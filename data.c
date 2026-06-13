/*This is our data generator. Totally overkill but overkill is on brand for someone attempting to patch a running binary's machine code from a child process to implement gradient descent. But the data becomes a permanent resident of your disk and can be inspected later. It won't pay rent though. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N_DATA 100
#define NOISE_SIGMA 0.05
static double randn() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}
// Nothing much to explain for the above helper, just that I love Box-Muller
int main() {
    srand(42);
    double xs[N_DATA];
    double ys[N_DATA];
    for (int i = 0; i < N_DATA; i++) {
        xs[i] = -M_PI + (2.0 * M_PI * i) / (N_DATA - 1);
        ys[i] = sin(xs[i]) + randn() * NOISE_SIGMA;
    }
    FILE *f = fopen("data.bin", "wb");
    if (!f) { perror("fopen"); return 1; }
    fwrite(xs, sizeof(double), N_DATA, f);
    fwrite(ys, sizeof(double), N_DATA, f);
    fclose(f);
    printf("wrote %d points to data.bin\n", N_DATA);
    return 0;
}
