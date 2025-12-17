use anyhow::Result;
use clap::Parser;
use jpgfromraw::{extract_directory, ExtractionConfig};
use std::ffi::OsString;
use std::path::PathBuf;

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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let config = ExtractionConfig::new(args.input_dir, args.output_dir)
        .with_transfers(args.transfers)
        .with_extension(args.extension);

    extract_directory(config).await
}
