param(
    [ValidateSet('core', 'enhanced', 'all')]
    [string]$Suite = 'core'
)

$ErrorActionPreference = 'Stop'
$python = 'C:\Users\yaoli\anaconda3\envs\thesis\python.exe'
$runner = 'D:\thesis\code\training\run_opensource_da_48_54_benchmark.py'
$reporter = 'D:\thesis\code\training\build_opensource_requested_scope_summary.py'
$logDir = 'D:\thesis\tables\opensource_da_48_54'
$log = Join-Path $logDir 'overnight_training.log'

foreach ($required in @($python, $runner, $reporter)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Invoke-BenchmarkStage {
    param([string[]]$Arguments)
    & $python $runner @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark stage failed with exit code $LASTEXITCODE. See $log"
    }
}

if ($Suite -in @('core', 'all')) {
    # 54D: only the requested open-source DANN + end-to-end linear/LR head,
    # with source-only controls and all three representation choices.
    Invoke-BenchmarkStage -Arguments @(
        '--feature-sets', '54D',
        '--splits', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7',
        '--ml-models',
        '--methods', 'source-only', 'dann',
        '--representations', 'bottleneck16', 'residual-d', 'fusion16',
        '--heads', 'linear'
    )

    # 48D core: classical LR/MLP and all standard/project DA methods on the
    # shared 16D bottleneck, with both end-to-end classification heads.
    Invoke-BenchmarkStage -Arguments @(
        '--feature-sets', '48D',
        '--splits', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7',
        '--ml-models', 'logistic-regression', 'mlp',
        '--methods', 'source-only', 'dann', 'deep-coral', 'mk-mmd', 'c-dann', 'cdan',
                     'cdan+c-dann', 'mk-mmd+lc', 'mk-mmd+cdan', 'mk-mmd+lc+cdan',
        '--representations', 'bottleneck16',
        '--heads', 'linear', 'mlp'
    )
}

if ($Suite -in @('enhanced', 'all')) {
    # 48D feature-preservation enhancements for every DA method.
    Invoke-BenchmarkStage -Arguments @(
        '--feature-sets', '48D',
        '--splits', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7',
        '--ml-models',
        '--methods', 'source-only', 'dann', 'deep-coral', 'mk-mmd', 'c-dann', 'cdan',
                     'cdan+c-dann', 'mk-mmd+lc', 'mk-mmd+cdan', 'mk-mmd+lc+cdan',
        '--representations', 'residual-d', 'fusion16',
        '--heads', 'linear', 'mlp'
    )
}

& $python $reporter
if ($LASTEXITCODE -ne 0) {
    throw "Requested-scope report failed with exit code $LASTEXITCODE."
}
