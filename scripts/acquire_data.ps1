param([string]$Destination = "data/raw")
$ErrorActionPreference = "Stop"
$repo = "data/upstream-esci-code"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
if (-not (Test-Path "$repo/.git")) {
    git clone --depth 1 https://github.com/amazon-research/esci-code.git $repo
}
git -C $repo lfs install
git -C $repo lfs pull --include="shopping_queries_dataset/shopping_queries_dataset_examples.parquet,shopping_queries_dataset/shopping_queries_dataset_products.parquet"
$names = "shopping_queries_dataset_examples.parquet", "shopping_queries_dataset_products.parquet"
foreach ($name in $names) {
    $source = Join-Path $repo "shopping_queries_dataset/$name"
    if (-not (Test-Path $source)) { throw "Expected LFS file missing: $source. Run: git -C $repo lfs pull" }
    $head = [System.IO.File]::ReadAllBytes($source)[0..7]
    if ([Text.Encoding]::ASCII.GetString($head) -eq "version h") { throw "Git LFS pointer detected: $source. Run git lfs pull." }
    $target = Join-Path $Destination $name
    if (Test-Path $target) { throw "Refusing to overwrite existing raw file: $target" }
    Copy-Item -LiteralPath $source -Destination $target
}
Write-Host "Real raw data copied to $Destination. Raw files were not modified."
