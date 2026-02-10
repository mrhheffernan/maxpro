use clap::ValueEnum;

#[derive(ValueEnum, Clone, Debug)]
pub enum Metrics {
    MaxPro,
    MaxiMin,
}
