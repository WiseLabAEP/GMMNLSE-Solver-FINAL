__global__ void calculate_sumterm_part(float2 * Up, float2 * Vpl, const float2 * A_t, const float* SR, const unsigned char* nonzero_midx1234s, const unsigned int N, const unsigned int M, const float SK_factor, const unsigned int NUM_NONZERO, const unsigned int NUM_MODES) {
    unsigned int full_thread_idx = threadIdx.x + blockIdx.x*blockDim.x;

    // Calculate the index
    unsigned int Midx = full_thread_idx / N;
    unsigned int Nidx = full_thread_idx % N;
    unsigned int NM = N*M;

    if (full_thread_idx >= NM) {
        return;
    }

    // Compute the tensors
    for (int i = 0; i < NUM_NONZERO; i++) {
        unsigned int midx1 = nonzero_midx1234s[i*4]-1;
        unsigned int midx2 = nonzero_midx1234s[1+i*4]-1;
        unsigned int midx3 = nonzero_midx1234s[2+i*4]-1;
        unsigned int midx4 = nonzero_midx1234s[3+i*4]-1;

        float a = A_t[Nidx+Midx*N+midx2*NM].x;
        float b = A_t[Nidx+Midx*N+midx2*NM].y;
        float c = A_t[Nidx+Midx*N+midx3*NM].x;
        float d = A_t[Nidx+Midx*N+midx3*NM].y;
        float e = A_t[Nidx+Midx*N+midx4*NM].x;
        float f = A_t[Nidx+Midx*N+midx4*NM].y;

		
        Up[Nidx+Midx*N+midx1*NM].x = Up[Nidx+Midx*N+midx1*NM].x + SK_factor*SR[i]*(a*c*e-b*d*e+a*d*f+c*b*f);
        Up[Nidx+Midx*N+midx1*NM].y = Up[Nidx+Midx*N+midx1*NM].y + SK_factor*SR[i]*(a*d*e+c*b*e-a*c*f+b*d*f);
        Vpl[Nidx+Midx*N+midx1*NM+midx2*NM*NUM_MODES].x = Vpl[Nidx+Midx*N+midx1*NM+midx2*NM*NUM_MODES].x + SR[i]*(c*e+d*f);
        Vpl[Nidx+Midx*N+midx1*NM+midx2*NM*NUM_MODES].y = Vpl[Nidx+Midx*N+midx1*NM+midx2*NM*NUM_MODES].y + SR[i]*(d*e-c*f);
    }
}
