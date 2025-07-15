use anyhow::{bail, ensure, Result};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use std::collections::HashSet;
use std::ffi::OsString;
use std::io::IoSlice;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

#[cfg(unix)]
use unix as platform;
#[cfg(windows)]
use windows as platform;

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Input directory containing RAW files
    input_dir: PathBuf,

    /// Output directory to store extracted JPEGs
    #[arg(default_value = ".")]
    output_dir: PathBuf,

    /// How many files to process at once
    #[arg(short, long, default_value_t = 8)]
    transfers: usize,

    /// Look for this extension in addition to the default list.
    ///
    /// Default list: arw, cr2, crw, dng, erf, kdc, mef, mrw, nef, nrw, orf, pef, raf, raw, rw2,
    /// rwl, sr2, srf, srw, x3f
    #[arg(short, long)]
    extension: Option<OsString>,
}

/// An embedded JPEG in a RAW file.
#[derive(Default, Eq, PartialEq)]
struct EmbeddedJpegInfo {
    offset: usize,
    length: usize,
    orientation: Option<u16>,
}

fn find_largest_embedded_jpeg_impl<B: ByteOrder>(raw_buf: &[u8]) -> Result<EmbeddedJpegInfo> {
    const IFD_ENTRY_SIZE: usize = 12;
    const JPEG_TAG: u16 = 0x201;
    const JPEG_LENGTH_TAG: u16 = 0x202;
    const ORIENTATION_TAG: u16 = 0x112;

    let mut next_ifd_offset = B::read_u32(&raw_buf[4..8]).try_into()?;
    let mut largest_jpeg = EmbeddedJpegInfo::default();

    while next_ifd_offset != 0 {
        ensure!(next_ifd_offset + 2 <= raw_buf.len(), "Invalid IFD offset");

        let cursor = &raw_buf[next_ifd_offset..];
        let num_entries = B::read_u16(&cursor[..2]).into();
        let entries_cursor = &cursor[2..];

        let entries_len = num_entries * IFD_ENTRY_SIZE;
        ensure!(
            entries_cursor.len() >= entries_len,
            "Invalid number of IFD entries"
        );

        let mut cur_offset = None;
        let mut cur_length = None;
        let mut cur_orientation = None;

        for entry in entries_cursor
            .chunks_exact(IFD_ENTRY_SIZE)
            .take(num_entries)
        {
            let tag = B::read_u16(&entry[..2]);

            match tag {
                JPEG_TAG => cur_offset = Some(B::read_u32(&entry[8..12]).try_into()?),
                JPEG_LENGTH_TAG => cur_length = Some(B::read_u32(&entry[8..12]).try_into()?),
                ORIENTATION_TAG => cur_orientation = Some(B::read_u16(&entry[8..10])),
                _ => {}
            }
        }

        if let (Some(offset), Some(length)) = (cur_offset, cur_length) {
            if length > largest_jpeg.length {
                largest_jpeg = EmbeddedJpegInfo {
                    offset,
                    length,
                    orientation: cur_orientation,
                };
            }
        }

        let next_ifd_offset_offset = 2 + entries_len;
        ensure!(
            cursor.len() >= next_ifd_offset_offset + 4,
            "Invalid next IFD offset"
        );
        next_ifd_offset = B::read_u32(&cursor[next_ifd_offset_offset..][..4]).try_into()?;
    }

    Ok(largest_jpeg)
}

/// Find the largest embedded JPEG data in a memory-mapped RAW buffer.
///
/// This function parses the IFDs in the TIFF structure of the RAW file to find the largest JPEG
/// thumbnail embedded in the file.
///
/// We hand roll the IFD parsing because libraries do not fit requirements. For example:
///
/// - kamadak-exif: Reads into a big `Vec<u8>`, which is huge for our big RAW.
/// - quickexif: Cannot iterate over IFDs.
fn find_largest_embedded_jpeg(raw_buf: &[u8]) -> Result<EmbeddedJpegInfo> {
    const TIFF_MAGIC_LE: &[u8] = b"II*\0";
    const TIFF_MAGIC_BE: &[u8] = b"MM\0*";

    ensure!(raw_buf.len() >= 8, "Not enough data for TIFF header");

    let is_le = &raw_buf[0..4] == TIFF_MAGIC_LE;
    ensure!(
        is_le || &raw_buf[0..4] == TIFF_MAGIC_BE,
        "Not a valid TIFF file"
    );

    let largest_jpeg = if is_le {
        find_largest_embedded_jpeg_impl::<LittleEndian>(raw_buf)?
    } else {
        find_largest_embedded_jpeg_impl::<BigEndian>(raw_buf)?
    };

    ensure!(
        largest_jpeg != EmbeddedJpegInfo::default(),
        "No JPEG data found"
    );
    ensure!(
        largest_jpeg.offset + largest_jpeg.length <= raw_buf.len(),
        "JPEG data exceeds file size"
    );

    Ok(largest_jpeg)
}

/// Extract the JPEG bytes from the memory-mapped RAW buffer.
fn extract_jpeg<'raw>(raw_buf: &'raw Mmap, jpeg: &'raw EmbeddedJpegInfo) -> Result<&'raw [u8]> {
    platform::prefetch_jpeg(raw_buf, jpeg)?;
    Ok(&raw_buf[jpeg.offset..jpeg.offset + jpeg.length])
}

/// The embedded JPEG comes with no EXIF data. While most of it is outside of the scope of this
/// application, it's pretty vexing to have the wrong orientation, so copy that over.
#[rustfmt::skip]
const fn get_header_bytes(orientation: u16) -> [u8; 34] {
    let orientation_bytes = orientation.to_le_bytes();
    [
        0xff, 0xd8, // SOI
        0xff, 0xe1, // APP1
        0x00, 0x1e, // 30 bytes including this length
        0x45, 0x78, 0x69, 0x66, 0x00, 0x00, // Exif\0\0
        0x49, 0x49, 0x2A, 0x00, // TIFF LE
        0x08, 0x00, 0x00, 0x00, // Offset to IFD
        0x01, 0x00, // One entry in IFD
        0x12, 0x01, // Tag for orientation
        0x03, 0x00, // Type: SHORT
        0x01, 0x00, 0x00, 0x00, // Count: 1
        orientation_bytes[0], orientation_bytes[1], // Orientation
        0x00, 0x00, // Next IFD
    ]
}

async fn write_jpeg(
    output_file: &Path,
    jpeg_buf: &[u8],
    jpeg_info: &EmbeddedJpegInfo,
) -> Result<()> {
    let mut out_file = File::create(output_file).await?;
    let hdr_bytes = get_header_bytes(jpeg_info.orientation.unwrap_or(1)); // 1 is default
    let mut header = hdr_bytes.as_ref();
    let mut body = &jpeg_buf[2..];

    while header.len() + body.len() > 0 {
        let slices = [IoSlice::new(header), IoSlice::new(body)];
        let n = out_file.write_vectored(&slices).await?;
        ensure!(n > 0, "Vectored write failed");

        if n < header.len() {
            // We didn't get past the header yet, remove bytes from it
            header = &header[n..];
        } else {
            // We are past the header and are in the body, remove bytes from the body
            let n = n - header.len();
            header = &[];
            body = &body[n..];
        }
    }

    Ok(())
}

/// Process a single RAW file to extract the embedded JPEG, and then write the extracted JPEG to
/// the output directory.
async fn process_file(entry_path: &Path, out_dir: &Path, relative_path: &Path) -> Result<()> {
    let in_file = platform::open_raw(entry_path).await?;
    let raw_buf = Arc::new(platform::mmap_raw(in_file)?);

    let raw_buf_clone = raw_buf.clone();
    let jpeg_info =
        tokio::task::spawn_blocking(move || find_largest_embedded_jpeg(&raw_buf_clone)).await??;

    let jpeg_buf = extract_jpeg(&raw_buf, &jpeg_info)?;
    let mut output_file = out_dir.join(relative_path);
    output_file.set_extension("jpg");
    write_jpeg(&output_file, jpeg_buf, &jpeg_info).await?;
    Ok(())
}

struct ProcessingResult {
    result: Result<()>,
    path: PathBuf,
}

/// Recursively process a directory of RAW files, extracting embedded JPEGs and writing them to the
/// output directory.
///
/// This function recursively searches the input directory for RAW files with valid extensions,
/// processes each file to extract the embedded JPEG, and writes the JPEGs to the corresponding
/// location in the output directory. The directory structure relative to the input directory is
/// maintained.
async fn process_directory(
    in_dir: &Path,
    out_dir: &'static Path,
    ext: Option<OsString>,
    transfers: usize,
) -> Result<()> {
    let valid_extensions = [
        "arw", "cr2", "crw", "dng", "erf", "kdc", "mef", "mrw", "nef", "nrw", "orf", "pef", "raf",
        "raw", "rw2", "rwl", "sr2", "srf", "srw", "x3f",
    ]
    .iter()
    .flat_map(|&ext| [OsString::from(ext), OsString::from(ext.to_uppercase())])
    .chain(ext.into_iter())
    .collect::<HashSet<_>>();

    let mut entries = Vec::new();
    let mut dir_queue = vec![in_dir.to_path_buf()];

    while let Some(current_dir) = dir_queue.pop() {
        let mut read_dir = fs::read_dir(&current_dir).await?;
        let mut found_raw = false;

        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if entry.file_type().await?.is_dir() {
                dir_queue.push(path);
            } else if path
                .extension()
                .is_some_and(|ext| valid_extensions.contains(ext))
            {
                found_raw = true;
                entries.push(path);
            }
        }

        if found_raw {
            let relative_dir = current_dir.strip_prefix(in_dir)?;
            let output_subdir = out_dir.join(relative_dir);
            fs::create_dir_all(&output_subdir).await?;
        }
    }

    let progress_bar = ProgressBar::new(entries.len().try_into()?);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{pos}/{len} [{bar}] (ETA: {eta})")?
            .progress_chars("##-"),
    );

    let semaphore = Arc::new(Semaphore::new(transfers));
    let mut tasks = Vec::with_capacity(entries.len());

    for in_path in entries {
        let semaphore = semaphore.clone();
        let relative_path = in_path.strip_prefix(in_dir)?.to_path_buf();
        let progress_bar = progress_bar.clone();
        let task: tokio::task::JoinHandle<Result<ProcessingResult>> = tokio::spawn(async move {
            let permit = semaphore.acquire_owned().await?;
            let result = process_file(&in_path, out_dir, &relative_path).await;
            drop(permit);
            progress_bar.inc(1);
            Ok(ProcessingResult {
                result,
                path: in_path,
            })
        });
        tasks.push(task);
    }

    let mut nr_failed = 0;
    for task in tasks {
        let pr_res = task.await??;
        if let Err(e) = pr_res.result {
            nr_failed += 1;
            let msg = format!("Error processing file {}: {:?}", pr_res.path.display(), e);
            progress_bar.println(msg);
        }
    }

    progress_bar.abandon();

    if nr_failed != 0 {
        bail!("Failed to process {} files", nr_failed);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // We would need a copy for each task otherwise, so better just to make it &'static
    let output_dir = Box::leak(Box::new(args.output_dir));

    fs::create_dir_all(&output_dir).await?;
    process_directory(&args.input_dir, output_dir, args.extension, args.transfers).await?;

    Ok(())
}
