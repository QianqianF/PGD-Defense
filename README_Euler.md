# Uncertainty Projection

### Submitting a job on Euler
Preparing environment
```
cd PGD-Defense
source euler_setup.sh
```
Submitting a job
```
bsub -R "rusage[mem=10240,ngpus_excl_p=1]" -o <output filename> 'python command'

# -W for time eg -W 8:00 for 8 hours
# mem: CPU mem
# ngpus_excl_p: number of GPUs requested
```
More info on config options see [https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs](https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs)

Check job status
```
bjobs
```
See why job is still pending
```
bjobs -p
```
After the job starts running,
```
bpeek
```
to get real time std output.

When the bjobs output shows `No unfinished job found`, an output file will be created at `<output filename>`.

