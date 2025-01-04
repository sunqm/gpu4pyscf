#include <cuda.h>
#include "vhf.cuh"
#include "rys1_roots.cu"
#include "create_tasks.cu"


__device__ static
void _rys_jk_0000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(1, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 1; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(1, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 1; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+2*nsq_per_block;
                    rys_roots(1, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(1, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 1; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        gout0 += 1 * 1 * wt;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_jk_0000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_0000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(1, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 1; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(1, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 1; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+2*nsq_per_block;
                    rys_roots(1, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(1, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 1; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        gout0 += trr_10x * 1 * wt;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += 1 * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += 1 * 1 * trr_10z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_jk_1000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        gout0 += trr_11x * 1 * wt;
                        double trr_01x = cpx * 1;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += trr_01x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += trr_01x * 1 * trr_10z;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gout3 += trr_10x * trr_01y * wt;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        gout4 += 1 * trr_11y * wt;
                        gout5 += 1 * trr_01y * trr_10z;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout6 += trr_10x * 1 * trr_01z;
                        gout7 += 1 * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout8 += 1 * 1 * trr_11z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout6 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    val += gout7 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    val += gout8 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+0)];
                    val += gout4 * dm[(j0+0)*nao+(i0+1)];
                    val += gout5 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+0)*nao+(i0+1)];
                    val += gout8 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+1)];
                    val += gout6 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+0)*nao+(k0+1)];
                    val += gout7 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout3 * dm[(i0+0)*nao+(k0+1)];
                    val += gout6 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout4 * dm[(i0+1)*nao+(k0+1)];
                    val += gout7 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout5 * dm[(i0+2)*nao+(k0+1)];
                    val += gout8 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+1)*nao+(l0+0)];
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_jk_1010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1010(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1011(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double gout18;
    double gout19;
    double gout20;
    double gout21;
    double gout22;
    double gout23;
    double gout24;
    double gout25;
    double gout26;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        gout18 = 0;
        gout19 = 0;
        gout20 = 0;
        gout21 = 0;
        gout22 = 0;
        gout23 = 0;
        gout24 = 0;
        gout25 = 0;
        gout26 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double b01 = .5/akl * (1 - rt_akl);
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_01x = cpx * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        gout0 += hrr_1011x * 1 * wt;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += hrr_0011x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += hrr_0011x * 1 * trr_10z;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gout3 += hrr_1001x * trr_01y * wt;
                        double hrr_0001x = trr_01x - xlxk * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        gout4 += hrr_0001x * trr_11y * wt;
                        gout5 += hrr_0001x * trr_01y * trr_10z;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout6 += hrr_1001x * 1 * trr_01z;
                        gout7 += hrr_0001x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout8 += hrr_0001x * 1 * trr_11z;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        gout9 += trr_11x * hrr_0001y * wt;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gout10 += trr_01x * hrr_1001y * wt;
                        gout11 += trr_01x * hrr_0001y * trr_10z;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        gout12 += trr_10x * hrr_0011y * wt;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        gout13 += 1 * hrr_1011y * wt;
                        gout14 += 1 * hrr_0011y * trr_10z;
                        gout15 += trr_10x * hrr_0001y * trr_01z;
                        gout16 += 1 * hrr_1001y * trr_01z;
                        gout17 += 1 * hrr_0001y * trr_11z;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        gout18 += trr_11x * 1 * hrr_0001z;
                        gout19 += trr_01x * trr_10y * hrr_0001z;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gout20 += trr_01x * 1 * hrr_1001z;
                        gout21 += trr_10x * trr_01y * hrr_0001z;
                        gout22 += 1 * trr_11y * hrr_0001z;
                        gout23 += 1 * trr_01y * hrr_1001z;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        gout24 += trr_10x * 1 * hrr_0011z;
                        gout25 += 1 * trr_10y * hrr_0011z;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        gout26 += 1 * 1 * hrr_1011z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout6 * dm[(l0+0)*nao+(k0+2)];
                    val += gout9 * dm[(l0+1)*nao+(k0+0)];
                    val += gout12 * dm[(l0+1)*nao+(k0+1)];
                    val += gout15 * dm[(l0+1)*nao+(k0+2)];
                    val += gout18 * dm[(l0+2)*nao+(k0+0)];
                    val += gout21 * dm[(l0+2)*nao+(k0+1)];
                    val += gout24 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    val += gout7 * dm[(l0+0)*nao+(k0+2)];
                    val += gout10 * dm[(l0+1)*nao+(k0+0)];
                    val += gout13 * dm[(l0+1)*nao+(k0+1)];
                    val += gout16 * dm[(l0+1)*nao+(k0+2)];
                    val += gout19 * dm[(l0+2)*nao+(k0+0)];
                    val += gout22 * dm[(l0+2)*nao+(k0+1)];
                    val += gout25 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    val += gout8 * dm[(l0+0)*nao+(k0+2)];
                    val += gout11 * dm[(l0+1)*nao+(k0+0)];
                    val += gout14 * dm[(l0+1)*nao+(k0+1)];
                    val += gout17 * dm[(l0+1)*nao+(k0+2)];
                    val += gout20 * dm[(l0+2)*nao+(k0+0)];
                    val += gout23 * dm[(l0+2)*nao+(k0+1)];
                    val += gout26 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+0)];
                    val += gout4 * dm[(j0+0)*nao+(i0+1)];
                    val += gout5 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+0)*nao+(i0+1)];
                    val += gout8 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+0)];
                    val += gout10 * dm[(j0+0)*nao+(i0+1)];
                    val += gout11 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+0)];
                    val += gout13 * dm[(j0+0)*nao+(i0+1)];
                    val += gout14 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(i0+0)];
                    val += gout16 * dm[(j0+0)*nao+(i0+1)];
                    val += gout17 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(i0+0)];
                    val += gout19 * dm[(j0+0)*nao+(i0+1)];
                    val += gout20 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout21 * dm[(j0+0)*nao+(i0+0)];
                    val += gout22 * dm[(j0+0)*nao+(i0+1)];
                    val += gout23 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout24 * dm[(j0+0)*nao+(i0+0)];
                    val += gout25 * dm[(j0+0)*nao+(i0+1)];
                    val += gout26 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+1)];
                    val += gout6 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    val += gout12 * dm[(j0+0)*nao+(k0+1)];
                    val += gout15 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(k0+0)];
                    val += gout21 * dm[(j0+0)*nao+(k0+1)];
                    val += gout24 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+0)*nao+(k0+1)];
                    val += gout7 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(k0+0)];
                    val += gout13 * dm[(j0+0)*nao+(k0+1)];
                    val += gout16 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout19 * dm[(j0+0)*nao+(k0+0)];
                    val += gout22 * dm[(j0+0)*nao+(k0+1)];
                    val += gout25 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(k0+0)];
                    val += gout14 * dm[(j0+0)*nao+(k0+1)];
                    val += gout17 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout20 * dm[(j0+0)*nao+(k0+0)];
                    val += gout23 * dm[(j0+0)*nao+(k0+1)];
                    val += gout26 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout3 * dm[(i0+0)*nao+(k0+1)];
                    val += gout6 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout4 * dm[(i0+1)*nao+(k0+1)];
                    val += gout7 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout5 * dm[(i0+2)*nao+(k0+1)];
                    val += gout8 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(k0+0)];
                    val += gout12 * dm[(i0+0)*nao+(k0+1)];
                    val += gout15 * dm[(i0+0)*nao+(k0+2)];
                    val += gout10 * dm[(i0+1)*nao+(k0+0)];
                    val += gout13 * dm[(i0+1)*nao+(k0+1)];
                    val += gout16 * dm[(i0+1)*nao+(k0+2)];
                    val += gout11 * dm[(i0+2)*nao+(k0+0)];
                    val += gout14 * dm[(i0+2)*nao+(k0+1)];
                    val += gout17 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(k0+0)];
                    val += gout21 * dm[(i0+0)*nao+(k0+1)];
                    val += gout24 * dm[(i0+0)*nao+(k0+2)];
                    val += gout19 * dm[(i0+1)*nao+(k0+0)];
                    val += gout22 * dm[(i0+1)*nao+(k0+1)];
                    val += gout25 * dm[(i0+1)*nao+(k0+2)];
                    val += gout20 * dm[(i0+2)*nao+(k0+0)];
                    val += gout23 * dm[(i0+2)*nao+(k0+1)];
                    val += gout26 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+0)*nao+(l0+1)];
                    val += gout18 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+0)*nao+(l0+1)];
                    val += gout21 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    val += gout15 * dm[(j0+0)*nao+(l0+1)];
                    val += gout24 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+0)*nao+(l0+1)];
                    val += gout19 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout13 * dm[(j0+0)*nao+(l0+1)];
                    val += gout22 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    val += gout16 * dm[(j0+0)*nao+(l0+1)];
                    val += gout25 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+0)*nao+(l0+1)];
                    val += gout20 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout14 * dm[(j0+0)*nao+(l0+1)];
                    val += gout23 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    val += gout17 * dm[(j0+0)*nao+(l0+1)];
                    val += gout26 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout9 * dm[(i0+0)*nao+(l0+1)];
                    val += gout18 * dm[(i0+0)*nao+(l0+2)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout10 * dm[(i0+1)*nao+(l0+1)];
                    val += gout19 * dm[(i0+1)*nao+(l0+2)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout11 * dm[(i0+2)*nao+(l0+1)];
                    val += gout20 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout12 * dm[(i0+0)*nao+(l0+1)];
                    val += gout21 * dm[(i0+0)*nao+(l0+2)];
                    val += gout4 * dm[(i0+1)*nao+(l0+0)];
                    val += gout13 * dm[(i0+1)*nao+(l0+1)];
                    val += gout22 * dm[(i0+1)*nao+(l0+2)];
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    val += gout14 * dm[(i0+2)*nao+(l0+1)];
                    val += gout23 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout15 * dm[(i0+0)*nao+(l0+1)];
                    val += gout24 * dm[(i0+0)*nao+(l0+2)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout16 * dm[(i0+1)*nao+(l0+1)];
                    val += gout25 * dm[(i0+1)*nao+(l0+2)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    val += gout17 * dm[(i0+2)*nao+(l0+1)];
                    val += gout26 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_1011(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1011(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        gout0 += hrr_1100x * 1 * wt;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += hrr_0100x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += hrr_0100x * 1 * trr_10z;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gout3 += trr_10x * hrr_0100y * wt;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout4 += 1 * hrr_1100y * wt;
                        gout5 += 1 * hrr_0100y * trr_10z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout6 += trr_10x * 1 * hrr_0100z;
                        gout7 += 1 * trr_10y * hrr_0100z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout8 += 1 * 1 * hrr_1100z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+1)*nao+(i0+0)];
                    val += gout4 * dm[(j0+1)*nao+(i0+1)];
                    val += gout5 * dm[(j0+1)*nao+(i0+2)];
                    val += gout6 * dm[(j0+2)*nao+(i0+0)];
                    val += gout7 * dm[(j0+2)*nao+(i0+1)];
                    val += gout8 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+1)*nao+(k0+0)];
                    val += gout6 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+1)*nao+(k0+0)];
                    val += gout7 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+1)*nao+(k0+0)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(k0+0)];
                    val += gout4 * dm[(i0+1)*nao+(k0+0)];
                    val += gout5 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+0)];
                    val += gout7 * dm[(i0+1)*nao+(k0+0)];
                    val += gout8 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout4 * dm[(j0+1)*nao+(l0+0)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+1)*nao+(l0+0)];
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_jk_1100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1100(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1110(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double gout18;
    double gout19;
    double gout20;
    double gout21;
    double gout22;
    double gout23;
    double gout24;
    double gout25;
    double gout26;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        gout18 = 0;
        gout19 = 0;
        gout20 = 0;
        gout21 = 0;
        gout22 = 0;
        gout23 = 0;
        gout24 = 0;
        gout25 = 0;
        gout26 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        gout0 += hrr_1110x * 1 * wt;
                        double trr_01x = cpx * 1;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += hrr_0110x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += hrr_0110x * 1 * trr_10z;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gout3 += trr_11x * hrr_0100y * wt;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout4 += trr_01x * hrr_1100y * wt;
                        gout5 += trr_01x * hrr_0100y * trr_10z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout6 += trr_11x * 1 * hrr_0100z;
                        gout7 += trr_01x * trr_10y * hrr_0100z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout8 += trr_01x * 1 * hrr_1100z;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gout9 += hrr_1100x * trr_01y * wt;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        gout10 += hrr_0100x * trr_11y * wt;
                        gout11 += hrr_0100x * trr_01y * trr_10z;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gout12 += trr_10x * hrr_0110y * wt;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gout13 += 1 * hrr_1110y * wt;
                        gout14 += 1 * hrr_0110y * trr_10z;
                        gout15 += trr_10x * trr_01y * hrr_0100z;
                        gout16 += 1 * trr_11y * hrr_0100z;
                        gout17 += 1 * trr_01y * hrr_1100z;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout18 += hrr_1100x * 1 * trr_01z;
                        gout19 += hrr_0100x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout20 += hrr_0100x * 1 * trr_11z;
                        gout21 += trr_10x * hrr_0100y * trr_01z;
                        gout22 += 1 * hrr_1100y * trr_01z;
                        gout23 += 1 * hrr_0100y * trr_11z;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gout24 += trr_10x * 1 * hrr_0110z;
                        gout25 += 1 * trr_10y * hrr_0110z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gout26 += 1 * 1 * hrr_1110z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+1)];
                    val += gout18 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout12 * dm[(l0+0)*nao+(k0+1)];
                    val += gout21 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    val += gout15 * dm[(l0+0)*nao+(k0+1)];
                    val += gout24 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+1)];
                    val += gout19 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout13 * dm[(l0+0)*nao+(k0+1)];
                    val += gout22 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    val += gout16 * dm[(l0+0)*nao+(k0+1)];
                    val += gout25 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+1)];
                    val += gout20 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    val += gout14 * dm[(l0+0)*nao+(k0+1)];
                    val += gout23 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    val += gout17 * dm[(l0+0)*nao+(k0+1)];
                    val += gout26 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+1)*nao+(i0+0)];
                    val += gout4 * dm[(j0+1)*nao+(i0+1)];
                    val += gout5 * dm[(j0+1)*nao+(i0+2)];
                    val += gout6 * dm[(j0+2)*nao+(i0+0)];
                    val += gout7 * dm[(j0+2)*nao+(i0+1)];
                    val += gout8 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+0)];
                    val += gout10 * dm[(j0+0)*nao+(i0+1)];
                    val += gout11 * dm[(j0+0)*nao+(i0+2)];
                    val += gout12 * dm[(j0+1)*nao+(i0+0)];
                    val += gout13 * dm[(j0+1)*nao+(i0+1)];
                    val += gout14 * dm[(j0+1)*nao+(i0+2)];
                    val += gout15 * dm[(j0+2)*nao+(i0+0)];
                    val += gout16 * dm[(j0+2)*nao+(i0+1)];
                    val += gout17 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(i0+0)];
                    val += gout19 * dm[(j0+0)*nao+(i0+1)];
                    val += gout20 * dm[(j0+0)*nao+(i0+2)];
                    val += gout21 * dm[(j0+1)*nao+(i0+0)];
                    val += gout22 * dm[(j0+1)*nao+(i0+1)];
                    val += gout23 * dm[(j0+1)*nao+(i0+2)];
                    val += gout24 * dm[(j0+2)*nao+(i0+0)];
                    val += gout25 * dm[(j0+2)*nao+(i0+1)];
                    val += gout26 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+1)];
                    val += gout18 * dm[(j0+0)*nao+(k0+2)];
                    val += gout3 * dm[(j0+1)*nao+(k0+0)];
                    val += gout12 * dm[(j0+1)*nao+(k0+1)];
                    val += gout21 * dm[(j0+1)*nao+(k0+2)];
                    val += gout6 * dm[(j0+2)*nao+(k0+0)];
                    val += gout15 * dm[(j0+2)*nao+(k0+1)];
                    val += gout24 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+0)*nao+(k0+1)];
                    val += gout19 * dm[(j0+0)*nao+(k0+2)];
                    val += gout4 * dm[(j0+1)*nao+(k0+0)];
                    val += gout13 * dm[(j0+1)*nao+(k0+1)];
                    val += gout22 * dm[(j0+1)*nao+(k0+2)];
                    val += gout7 * dm[(j0+2)*nao+(k0+0)];
                    val += gout16 * dm[(j0+2)*nao+(k0+1)];
                    val += gout25 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    val += gout20 * dm[(j0+0)*nao+(k0+2)];
                    val += gout5 * dm[(j0+1)*nao+(k0+0)];
                    val += gout14 * dm[(j0+1)*nao+(k0+1)];
                    val += gout23 * dm[(j0+1)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    val += gout17 * dm[(j0+2)*nao+(k0+1)];
                    val += gout26 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout9 * dm[(i0+0)*nao+(k0+1)];
                    val += gout18 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout10 * dm[(i0+1)*nao+(k0+1)];
                    val += gout19 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout11 * dm[(i0+2)*nao+(k0+1)];
                    val += gout20 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(k0+0)];
                    val += gout12 * dm[(i0+0)*nao+(k0+1)];
                    val += gout21 * dm[(i0+0)*nao+(k0+2)];
                    val += gout4 * dm[(i0+1)*nao+(k0+0)];
                    val += gout13 * dm[(i0+1)*nao+(k0+1)];
                    val += gout22 * dm[(i0+1)*nao+(k0+2)];
                    val += gout5 * dm[(i0+2)*nao+(k0+0)];
                    val += gout14 * dm[(i0+2)*nao+(k0+1)];
                    val += gout23 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+0)];
                    val += gout15 * dm[(i0+0)*nao+(k0+1)];
                    val += gout24 * dm[(i0+0)*nao+(k0+2)];
                    val += gout7 * dm[(i0+1)*nao+(k0+0)];
                    val += gout16 * dm[(i0+1)*nao+(k0+1)];
                    val += gout25 * dm[(i0+1)*nao+(k0+2)];
                    val += gout8 * dm[(i0+2)*nao+(k0+0)];
                    val += gout17 * dm[(i0+2)*nao+(k0+1)];
                    val += gout26 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+1)*nao+(l0+0)];
                    val += gout15 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(l0+0)];
                    val += gout21 * dm[(j0+1)*nao+(l0+0)];
                    val += gout24 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout4 * dm[(j0+1)*nao+(l0+0)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    val += gout13 * dm[(j0+1)*nao+(l0+0)];
                    val += gout16 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout19 * dm[(j0+0)*nao+(l0+0)];
                    val += gout22 * dm[(j0+1)*nao+(l0+0)];
                    val += gout25 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+0)];
                    val += gout14 * dm[(j0+1)*nao+(l0+0)];
                    val += gout17 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout20 * dm[(j0+0)*nao+(l0+0)];
                    val += gout23 * dm[(j0+1)*nao+(l0+0)];
                    val += gout26 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(l0+0)];
                    val += gout10 * dm[(i0+1)*nao+(l0+0)];
                    val += gout11 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(l0+0)];
                    val += gout19 * dm[(i0+1)*nao+(l0+0)];
                    val += gout20 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+1)*nao+(l0+0)];
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+1)*nao+(l0+0)];
                    val += gout14 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout21 * dm[(i0+0)*nao+(l0+0)];
                    val += gout22 * dm[(i0+1)*nao+(l0+0)];
                    val += gout23 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout15 * dm[(i0+0)*nao+(l0+0)];
                    val += gout16 * dm[(i0+1)*nao+(l0+0)];
                    val += gout17 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout24 * dm[(i0+0)*nao+(l0+0)];
                    val += gout25 * dm[(i0+1)*nao+(l0+0)];
                    val += gout26 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_1110(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1110(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1111(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double gout18;
    double gout19;
    double gout20;
    double gout21;
    double gout22;
    double gout23;
    double gout24;
    double gout25;
    double gout26;
    double gout27;
    double gout28;
    double gout29;
    double gout30;
    double gout31;
    double gout32;
    double gout33;
    double gout34;
    double gout35;
    double gout36;
    double gout37;
    double gout38;
    double gout39;
    double gout40;
    double gout41;
    double gout42;
    double gout43;
    double gout44;
    double gout45;
    double gout46;
    double gout47;
    double gout48;
    double gout49;
    double gout50;
    double gout51;
    double gout52;
    double gout53;
    double gout54;
    double gout55;
    double gout56;
    double gout57;
    double gout58;
    double gout59;
    double gout60;
    double gout61;
    double gout62;
    double gout63;
    double gout64;
    double gout65;
    double gout66;
    double gout67;
    double gout68;
    double gout69;
    double gout70;
    double gout71;
    double gout72;
    double gout73;
    double gout74;
    double gout75;
    double gout76;
    double gout77;
    double gout78;
    double gout79;
    double gout80;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        gout18 = 0;
        gout19 = 0;
        gout20 = 0;
        gout21 = 0;
        gout22 = 0;
        gout23 = 0;
        gout24 = 0;
        gout25 = 0;
        gout26 = 0;
        gout27 = 0;
        gout28 = 0;
        gout29 = 0;
        gout30 = 0;
        gout31 = 0;
        gout32 = 0;
        gout33 = 0;
        gout34 = 0;
        gout35 = 0;
        gout36 = 0;
        gout37 = 0;
        gout38 = 0;
        gout39 = 0;
        gout40 = 0;
        gout41 = 0;
        gout42 = 0;
        gout43 = 0;
        gout44 = 0;
        gout45 = 0;
        gout46 = 0;
        gout47 = 0;
        gout48 = 0;
        gout49 = 0;
        gout50 = 0;
        gout51 = 0;
        gout52 = 0;
        gout53 = 0;
        gout54 = 0;
        gout55 = 0;
        gout56 = 0;
        gout57 = 0;
        gout58 = 0;
        gout59 = 0;
        gout60 = 0;
        gout61 = 0;
        gout62 = 0;
        gout63 = 0;
        gout64 = 0;
        gout65 = 0;
        gout66 = 0;
        gout67 = 0;
        gout68 = 0;
        gout69 = 0;
        gout70 = 0;
        gout71 = 0;
        gout72 = 0;
        gout73 = 0;
        gout74 = 0;
        gout75 = 0;
        gout76 = 0;
        gout77 = 0;
        gout78 = 0;
        gout79 = 0;
        gout80 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double b01 = .5/akl * (1 - rt_akl);
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double hrr_2011x = trr_22x - xlxk * trr_21x;
                        double trr_01x = cpx * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                        gout0 += hrr_1111x * 1 * wt;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += hrr_0111x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += hrr_0111x * 1 * trr_10z;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gout3 += hrr_1011x * hrr_0100y * wt;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout4 += hrr_0011x * hrr_1100y * wt;
                        gout5 += hrr_0011x * hrr_0100y * trr_10z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout6 += hrr_1011x * 1 * hrr_0100z;
                        gout7 += hrr_0011x * trr_10y * hrr_0100z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout8 += hrr_0011x * 1 * hrr_1100z;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gout9 += hrr_1101x * trr_01y * wt;
                        double hrr_0001x = trr_01x - xlxk * 1;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        gout10 += hrr_0101x * trr_11y * wt;
                        gout11 += hrr_0101x * trr_01y * trr_10z;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gout12 += hrr_1001x * hrr_0110y * wt;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gout13 += hrr_0001x * hrr_1110y * wt;
                        gout14 += hrr_0001x * hrr_0110y * trr_10z;
                        gout15 += hrr_1001x * trr_01y * hrr_0100z;
                        gout16 += hrr_0001x * trr_11y * hrr_0100z;
                        gout17 += hrr_0001x * trr_01y * hrr_1100z;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout18 += hrr_1101x * 1 * trr_01z;
                        gout19 += hrr_0101x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout20 += hrr_0101x * 1 * trr_11z;
                        gout21 += hrr_1001x * hrr_0100y * trr_01z;
                        gout22 += hrr_0001x * hrr_1100y * trr_01z;
                        gout23 += hrr_0001x * hrr_0100y * trr_11z;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gout24 += hrr_1001x * 1 * hrr_0110z;
                        gout25 += hrr_0001x * trr_10y * hrr_0110z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gout26 += hrr_0001x * 1 * hrr_1110z;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        gout27 += hrr_1110x * hrr_0001y * wt;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gout28 += hrr_0110x * hrr_1001y * wt;
                        gout29 += hrr_0110x * hrr_0001y * trr_10z;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        gout30 += trr_11x * hrr_0101y * wt;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        gout31 += trr_01x * hrr_1101y * wt;
                        gout32 += trr_01x * hrr_0101y * trr_10z;
                        gout33 += trr_11x * hrr_0001y * hrr_0100z;
                        gout34 += trr_01x * hrr_1001y * hrr_0100z;
                        gout35 += trr_01x * hrr_0001y * hrr_1100z;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        gout36 += hrr_1100x * hrr_0011y * wt;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        gout37 += hrr_0100x * hrr_1011y * wt;
                        gout38 += hrr_0100x * hrr_0011y * trr_10z;
                        double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                        gout39 += trr_10x * hrr_0111y * wt;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                        gout40 += 1 * hrr_1111y * wt;
                        gout41 += 1 * hrr_0111y * trr_10z;
                        gout42 += trr_10x * hrr_0011y * hrr_0100z;
                        gout43 += 1 * hrr_1011y * hrr_0100z;
                        gout44 += 1 * hrr_0011y * hrr_1100z;
                        gout45 += hrr_1100x * hrr_0001y * trr_01z;
                        gout46 += hrr_0100x * hrr_1001y * trr_01z;
                        gout47 += hrr_0100x * hrr_0001y * trr_11z;
                        gout48 += trr_10x * hrr_0101y * trr_01z;
                        gout49 += 1 * hrr_1101y * trr_01z;
                        gout50 += 1 * hrr_0101y * trr_11z;
                        gout51 += trr_10x * hrr_0001y * hrr_0110z;
                        gout52 += 1 * hrr_1001y * hrr_0110z;
                        gout53 += 1 * hrr_0001y * hrr_1110z;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        gout54 += hrr_1110x * 1 * hrr_0001z;
                        gout55 += hrr_0110x * trr_10y * hrr_0001z;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gout56 += hrr_0110x * 1 * hrr_1001z;
                        gout57 += trr_11x * hrr_0100y * hrr_0001z;
                        gout58 += trr_01x * hrr_1100y * hrr_0001z;
                        gout59 += trr_01x * hrr_0100y * hrr_1001z;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        gout60 += trr_11x * 1 * hrr_0101z;
                        gout61 += trr_01x * trr_10y * hrr_0101z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        gout62 += trr_01x * 1 * hrr_1101z;
                        gout63 += hrr_1100x * trr_01y * hrr_0001z;
                        gout64 += hrr_0100x * trr_11y * hrr_0001z;
                        gout65 += hrr_0100x * trr_01y * hrr_1001z;
                        gout66 += trr_10x * hrr_0110y * hrr_0001z;
                        gout67 += 1 * hrr_1110y * hrr_0001z;
                        gout68 += 1 * hrr_0110y * hrr_1001z;
                        gout69 += trr_10x * trr_01y * hrr_0101z;
                        gout70 += 1 * trr_11y * hrr_0101z;
                        gout71 += 1 * trr_01y * hrr_1101z;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        gout72 += hrr_1100x * 1 * hrr_0011z;
                        gout73 += hrr_0100x * trr_10y * hrr_0011z;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        gout74 += hrr_0100x * 1 * hrr_1011z;
                        gout75 += trr_10x * hrr_0100y * hrr_0011z;
                        gout76 += 1 * hrr_1100y * hrr_0011z;
                        gout77 += 1 * hrr_0100y * hrr_1011z;
                        double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                        gout78 += trr_10x * 1 * hrr_0111z;
                        gout79 += 1 * trr_10y * hrr_0111z;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                        gout80 += 1 * 1 * hrr_1111z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+1)];
                    val += gout18 * dm[(l0+0)*nao+(k0+2)];
                    val += gout27 * dm[(l0+1)*nao+(k0+0)];
                    val += gout36 * dm[(l0+1)*nao+(k0+1)];
                    val += gout45 * dm[(l0+1)*nao+(k0+2)];
                    val += gout54 * dm[(l0+2)*nao+(k0+0)];
                    val += gout63 * dm[(l0+2)*nao+(k0+1)];
                    val += gout72 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout12 * dm[(l0+0)*nao+(k0+1)];
                    val += gout21 * dm[(l0+0)*nao+(k0+2)];
                    val += gout30 * dm[(l0+1)*nao+(k0+0)];
                    val += gout39 * dm[(l0+1)*nao+(k0+1)];
                    val += gout48 * dm[(l0+1)*nao+(k0+2)];
                    val += gout57 * dm[(l0+2)*nao+(k0+0)];
                    val += gout66 * dm[(l0+2)*nao+(k0+1)];
                    val += gout75 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    val += gout15 * dm[(l0+0)*nao+(k0+1)];
                    val += gout24 * dm[(l0+0)*nao+(k0+2)];
                    val += gout33 * dm[(l0+1)*nao+(k0+0)];
                    val += gout42 * dm[(l0+1)*nao+(k0+1)];
                    val += gout51 * dm[(l0+1)*nao+(k0+2)];
                    val += gout60 * dm[(l0+2)*nao+(k0+0)];
                    val += gout69 * dm[(l0+2)*nao+(k0+1)];
                    val += gout78 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+1)];
                    val += gout19 * dm[(l0+0)*nao+(k0+2)];
                    val += gout28 * dm[(l0+1)*nao+(k0+0)];
                    val += gout37 * dm[(l0+1)*nao+(k0+1)];
                    val += gout46 * dm[(l0+1)*nao+(k0+2)];
                    val += gout55 * dm[(l0+2)*nao+(k0+0)];
                    val += gout64 * dm[(l0+2)*nao+(k0+1)];
                    val += gout73 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout13 * dm[(l0+0)*nao+(k0+1)];
                    val += gout22 * dm[(l0+0)*nao+(k0+2)];
                    val += gout31 * dm[(l0+1)*nao+(k0+0)];
                    val += gout40 * dm[(l0+1)*nao+(k0+1)];
                    val += gout49 * dm[(l0+1)*nao+(k0+2)];
                    val += gout58 * dm[(l0+2)*nao+(k0+0)];
                    val += gout67 * dm[(l0+2)*nao+(k0+1)];
                    val += gout76 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    val += gout16 * dm[(l0+0)*nao+(k0+1)];
                    val += gout25 * dm[(l0+0)*nao+(k0+2)];
                    val += gout34 * dm[(l0+1)*nao+(k0+0)];
                    val += gout43 * dm[(l0+1)*nao+(k0+1)];
                    val += gout52 * dm[(l0+1)*nao+(k0+2)];
                    val += gout61 * dm[(l0+2)*nao+(k0+0)];
                    val += gout70 * dm[(l0+2)*nao+(k0+1)];
                    val += gout79 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+1)];
                    val += gout20 * dm[(l0+0)*nao+(k0+2)];
                    val += gout29 * dm[(l0+1)*nao+(k0+0)];
                    val += gout38 * dm[(l0+1)*nao+(k0+1)];
                    val += gout47 * dm[(l0+1)*nao+(k0+2)];
                    val += gout56 * dm[(l0+2)*nao+(k0+0)];
                    val += gout65 * dm[(l0+2)*nao+(k0+1)];
                    val += gout74 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    val += gout14 * dm[(l0+0)*nao+(k0+1)];
                    val += gout23 * dm[(l0+0)*nao+(k0+2)];
                    val += gout32 * dm[(l0+1)*nao+(k0+0)];
                    val += gout41 * dm[(l0+1)*nao+(k0+1)];
                    val += gout50 * dm[(l0+1)*nao+(k0+2)];
                    val += gout59 * dm[(l0+2)*nao+(k0+0)];
                    val += gout68 * dm[(l0+2)*nao+(k0+1)];
                    val += gout77 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    val += gout17 * dm[(l0+0)*nao+(k0+1)];
                    val += gout26 * dm[(l0+0)*nao+(k0+2)];
                    val += gout35 * dm[(l0+1)*nao+(k0+0)];
                    val += gout44 * dm[(l0+1)*nao+(k0+1)];
                    val += gout53 * dm[(l0+1)*nao+(k0+2)];
                    val += gout62 * dm[(l0+2)*nao+(k0+0)];
                    val += gout71 * dm[(l0+2)*nao+(k0+1)];
                    val += gout80 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+1)*nao+(i0+0)];
                    val += gout4 * dm[(j0+1)*nao+(i0+1)];
                    val += gout5 * dm[(j0+1)*nao+(i0+2)];
                    val += gout6 * dm[(j0+2)*nao+(i0+0)];
                    val += gout7 * dm[(j0+2)*nao+(i0+1)];
                    val += gout8 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+0)];
                    val += gout10 * dm[(j0+0)*nao+(i0+1)];
                    val += gout11 * dm[(j0+0)*nao+(i0+2)];
                    val += gout12 * dm[(j0+1)*nao+(i0+0)];
                    val += gout13 * dm[(j0+1)*nao+(i0+1)];
                    val += gout14 * dm[(j0+1)*nao+(i0+2)];
                    val += gout15 * dm[(j0+2)*nao+(i0+0)];
                    val += gout16 * dm[(j0+2)*nao+(i0+1)];
                    val += gout17 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(i0+0)];
                    val += gout19 * dm[(j0+0)*nao+(i0+1)];
                    val += gout20 * dm[(j0+0)*nao+(i0+2)];
                    val += gout21 * dm[(j0+1)*nao+(i0+0)];
                    val += gout22 * dm[(j0+1)*nao+(i0+1)];
                    val += gout23 * dm[(j0+1)*nao+(i0+2)];
                    val += gout24 * dm[(j0+2)*nao+(i0+0)];
                    val += gout25 * dm[(j0+2)*nao+(i0+1)];
                    val += gout26 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout27 * dm[(j0+0)*nao+(i0+0)];
                    val += gout28 * dm[(j0+0)*nao+(i0+1)];
                    val += gout29 * dm[(j0+0)*nao+(i0+2)];
                    val += gout30 * dm[(j0+1)*nao+(i0+0)];
                    val += gout31 * dm[(j0+1)*nao+(i0+1)];
                    val += gout32 * dm[(j0+1)*nao+(i0+2)];
                    val += gout33 * dm[(j0+2)*nao+(i0+0)];
                    val += gout34 * dm[(j0+2)*nao+(i0+1)];
                    val += gout35 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout36 * dm[(j0+0)*nao+(i0+0)];
                    val += gout37 * dm[(j0+0)*nao+(i0+1)];
                    val += gout38 * dm[(j0+0)*nao+(i0+2)];
                    val += gout39 * dm[(j0+1)*nao+(i0+0)];
                    val += gout40 * dm[(j0+1)*nao+(i0+1)];
                    val += gout41 * dm[(j0+1)*nao+(i0+2)];
                    val += gout42 * dm[(j0+2)*nao+(i0+0)];
                    val += gout43 * dm[(j0+2)*nao+(i0+1)];
                    val += gout44 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout45 * dm[(j0+0)*nao+(i0+0)];
                    val += gout46 * dm[(j0+0)*nao+(i0+1)];
                    val += gout47 * dm[(j0+0)*nao+(i0+2)];
                    val += gout48 * dm[(j0+1)*nao+(i0+0)];
                    val += gout49 * dm[(j0+1)*nao+(i0+1)];
                    val += gout50 * dm[(j0+1)*nao+(i0+2)];
                    val += gout51 * dm[(j0+2)*nao+(i0+0)];
                    val += gout52 * dm[(j0+2)*nao+(i0+1)];
                    val += gout53 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout54 * dm[(j0+0)*nao+(i0+0)];
                    val += gout55 * dm[(j0+0)*nao+(i0+1)];
                    val += gout56 * dm[(j0+0)*nao+(i0+2)];
                    val += gout57 * dm[(j0+1)*nao+(i0+0)];
                    val += gout58 * dm[(j0+1)*nao+(i0+1)];
                    val += gout59 * dm[(j0+1)*nao+(i0+2)];
                    val += gout60 * dm[(j0+2)*nao+(i0+0)];
                    val += gout61 * dm[(j0+2)*nao+(i0+1)];
                    val += gout62 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout63 * dm[(j0+0)*nao+(i0+0)];
                    val += gout64 * dm[(j0+0)*nao+(i0+1)];
                    val += gout65 * dm[(j0+0)*nao+(i0+2)];
                    val += gout66 * dm[(j0+1)*nao+(i0+0)];
                    val += gout67 * dm[(j0+1)*nao+(i0+1)];
                    val += gout68 * dm[(j0+1)*nao+(i0+2)];
                    val += gout69 * dm[(j0+2)*nao+(i0+0)];
                    val += gout70 * dm[(j0+2)*nao+(i0+1)];
                    val += gout71 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout72 * dm[(j0+0)*nao+(i0+0)];
                    val += gout73 * dm[(j0+0)*nao+(i0+1)];
                    val += gout74 * dm[(j0+0)*nao+(i0+2)];
                    val += gout75 * dm[(j0+1)*nao+(i0+0)];
                    val += gout76 * dm[(j0+1)*nao+(i0+1)];
                    val += gout77 * dm[(j0+1)*nao+(i0+2)];
                    val += gout78 * dm[(j0+2)*nao+(i0+0)];
                    val += gout79 * dm[(j0+2)*nao+(i0+1)];
                    val += gout80 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+1)];
                    val += gout18 * dm[(j0+0)*nao+(k0+2)];
                    val += gout3 * dm[(j0+1)*nao+(k0+0)];
                    val += gout12 * dm[(j0+1)*nao+(k0+1)];
                    val += gout21 * dm[(j0+1)*nao+(k0+2)];
                    val += gout6 * dm[(j0+2)*nao+(k0+0)];
                    val += gout15 * dm[(j0+2)*nao+(k0+1)];
                    val += gout24 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout27 * dm[(j0+0)*nao+(k0+0)];
                    val += gout36 * dm[(j0+0)*nao+(k0+1)];
                    val += gout45 * dm[(j0+0)*nao+(k0+2)];
                    val += gout30 * dm[(j0+1)*nao+(k0+0)];
                    val += gout39 * dm[(j0+1)*nao+(k0+1)];
                    val += gout48 * dm[(j0+1)*nao+(k0+2)];
                    val += gout33 * dm[(j0+2)*nao+(k0+0)];
                    val += gout42 * dm[(j0+2)*nao+(k0+1)];
                    val += gout51 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout54 * dm[(j0+0)*nao+(k0+0)];
                    val += gout63 * dm[(j0+0)*nao+(k0+1)];
                    val += gout72 * dm[(j0+0)*nao+(k0+2)];
                    val += gout57 * dm[(j0+1)*nao+(k0+0)];
                    val += gout66 * dm[(j0+1)*nao+(k0+1)];
                    val += gout75 * dm[(j0+1)*nao+(k0+2)];
                    val += gout60 * dm[(j0+2)*nao+(k0+0)];
                    val += gout69 * dm[(j0+2)*nao+(k0+1)];
                    val += gout78 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+0)*nao+(k0+1)];
                    val += gout19 * dm[(j0+0)*nao+(k0+2)];
                    val += gout4 * dm[(j0+1)*nao+(k0+0)];
                    val += gout13 * dm[(j0+1)*nao+(k0+1)];
                    val += gout22 * dm[(j0+1)*nao+(k0+2)];
                    val += gout7 * dm[(j0+2)*nao+(k0+0)];
                    val += gout16 * dm[(j0+2)*nao+(k0+1)];
                    val += gout25 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout28 * dm[(j0+0)*nao+(k0+0)];
                    val += gout37 * dm[(j0+0)*nao+(k0+1)];
                    val += gout46 * dm[(j0+0)*nao+(k0+2)];
                    val += gout31 * dm[(j0+1)*nao+(k0+0)];
                    val += gout40 * dm[(j0+1)*nao+(k0+1)];
                    val += gout49 * dm[(j0+1)*nao+(k0+2)];
                    val += gout34 * dm[(j0+2)*nao+(k0+0)];
                    val += gout43 * dm[(j0+2)*nao+(k0+1)];
                    val += gout52 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout55 * dm[(j0+0)*nao+(k0+0)];
                    val += gout64 * dm[(j0+0)*nao+(k0+1)];
                    val += gout73 * dm[(j0+0)*nao+(k0+2)];
                    val += gout58 * dm[(j0+1)*nao+(k0+0)];
                    val += gout67 * dm[(j0+1)*nao+(k0+1)];
                    val += gout76 * dm[(j0+1)*nao+(k0+2)];
                    val += gout61 * dm[(j0+2)*nao+(k0+0)];
                    val += gout70 * dm[(j0+2)*nao+(k0+1)];
                    val += gout79 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    val += gout20 * dm[(j0+0)*nao+(k0+2)];
                    val += gout5 * dm[(j0+1)*nao+(k0+0)];
                    val += gout14 * dm[(j0+1)*nao+(k0+1)];
                    val += gout23 * dm[(j0+1)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    val += gout17 * dm[(j0+2)*nao+(k0+1)];
                    val += gout26 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout29 * dm[(j0+0)*nao+(k0+0)];
                    val += gout38 * dm[(j0+0)*nao+(k0+1)];
                    val += gout47 * dm[(j0+0)*nao+(k0+2)];
                    val += gout32 * dm[(j0+1)*nao+(k0+0)];
                    val += gout41 * dm[(j0+1)*nao+(k0+1)];
                    val += gout50 * dm[(j0+1)*nao+(k0+2)];
                    val += gout35 * dm[(j0+2)*nao+(k0+0)];
                    val += gout44 * dm[(j0+2)*nao+(k0+1)];
                    val += gout53 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout56 * dm[(j0+0)*nao+(k0+0)];
                    val += gout65 * dm[(j0+0)*nao+(k0+1)];
                    val += gout74 * dm[(j0+0)*nao+(k0+2)];
                    val += gout59 * dm[(j0+1)*nao+(k0+0)];
                    val += gout68 * dm[(j0+1)*nao+(k0+1)];
                    val += gout77 * dm[(j0+1)*nao+(k0+2)];
                    val += gout62 * dm[(j0+2)*nao+(k0+0)];
                    val += gout71 * dm[(j0+2)*nao+(k0+1)];
                    val += gout80 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout9 * dm[(i0+0)*nao+(k0+1)];
                    val += gout18 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout10 * dm[(i0+1)*nao+(k0+1)];
                    val += gout19 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout11 * dm[(i0+2)*nao+(k0+1)];
                    val += gout20 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout27 * dm[(i0+0)*nao+(k0+0)];
                    val += gout36 * dm[(i0+0)*nao+(k0+1)];
                    val += gout45 * dm[(i0+0)*nao+(k0+2)];
                    val += gout28 * dm[(i0+1)*nao+(k0+0)];
                    val += gout37 * dm[(i0+1)*nao+(k0+1)];
                    val += gout46 * dm[(i0+1)*nao+(k0+2)];
                    val += gout29 * dm[(i0+2)*nao+(k0+0)];
                    val += gout38 * dm[(i0+2)*nao+(k0+1)];
                    val += gout47 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout54 * dm[(i0+0)*nao+(k0+0)];
                    val += gout63 * dm[(i0+0)*nao+(k0+1)];
                    val += gout72 * dm[(i0+0)*nao+(k0+2)];
                    val += gout55 * dm[(i0+1)*nao+(k0+0)];
                    val += gout64 * dm[(i0+1)*nao+(k0+1)];
                    val += gout73 * dm[(i0+1)*nao+(k0+2)];
                    val += gout56 * dm[(i0+2)*nao+(k0+0)];
                    val += gout65 * dm[(i0+2)*nao+(k0+1)];
                    val += gout74 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(k0+0)];
                    val += gout12 * dm[(i0+0)*nao+(k0+1)];
                    val += gout21 * dm[(i0+0)*nao+(k0+2)];
                    val += gout4 * dm[(i0+1)*nao+(k0+0)];
                    val += gout13 * dm[(i0+1)*nao+(k0+1)];
                    val += gout22 * dm[(i0+1)*nao+(k0+2)];
                    val += gout5 * dm[(i0+2)*nao+(k0+0)];
                    val += gout14 * dm[(i0+2)*nao+(k0+1)];
                    val += gout23 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout30 * dm[(i0+0)*nao+(k0+0)];
                    val += gout39 * dm[(i0+0)*nao+(k0+1)];
                    val += gout48 * dm[(i0+0)*nao+(k0+2)];
                    val += gout31 * dm[(i0+1)*nao+(k0+0)];
                    val += gout40 * dm[(i0+1)*nao+(k0+1)];
                    val += gout49 * dm[(i0+1)*nao+(k0+2)];
                    val += gout32 * dm[(i0+2)*nao+(k0+0)];
                    val += gout41 * dm[(i0+2)*nao+(k0+1)];
                    val += gout50 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout57 * dm[(i0+0)*nao+(k0+0)];
                    val += gout66 * dm[(i0+0)*nao+(k0+1)];
                    val += gout75 * dm[(i0+0)*nao+(k0+2)];
                    val += gout58 * dm[(i0+1)*nao+(k0+0)];
                    val += gout67 * dm[(i0+1)*nao+(k0+1)];
                    val += gout76 * dm[(i0+1)*nao+(k0+2)];
                    val += gout59 * dm[(i0+2)*nao+(k0+0)];
                    val += gout68 * dm[(i0+2)*nao+(k0+1)];
                    val += gout77 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+0)];
                    val += gout15 * dm[(i0+0)*nao+(k0+1)];
                    val += gout24 * dm[(i0+0)*nao+(k0+2)];
                    val += gout7 * dm[(i0+1)*nao+(k0+0)];
                    val += gout16 * dm[(i0+1)*nao+(k0+1)];
                    val += gout25 * dm[(i0+1)*nao+(k0+2)];
                    val += gout8 * dm[(i0+2)*nao+(k0+0)];
                    val += gout17 * dm[(i0+2)*nao+(k0+1)];
                    val += gout26 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout33 * dm[(i0+0)*nao+(k0+0)];
                    val += gout42 * dm[(i0+0)*nao+(k0+1)];
                    val += gout51 * dm[(i0+0)*nao+(k0+2)];
                    val += gout34 * dm[(i0+1)*nao+(k0+0)];
                    val += gout43 * dm[(i0+1)*nao+(k0+1)];
                    val += gout52 * dm[(i0+1)*nao+(k0+2)];
                    val += gout35 * dm[(i0+2)*nao+(k0+0)];
                    val += gout44 * dm[(i0+2)*nao+(k0+1)];
                    val += gout53 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout60 * dm[(i0+0)*nao+(k0+0)];
                    val += gout69 * dm[(i0+0)*nao+(k0+1)];
                    val += gout78 * dm[(i0+0)*nao+(k0+2)];
                    val += gout61 * dm[(i0+1)*nao+(k0+0)];
                    val += gout70 * dm[(i0+1)*nao+(k0+1)];
                    val += gout79 * dm[(i0+1)*nao+(k0+2)];
                    val += gout62 * dm[(i0+2)*nao+(k0+0)];
                    val += gout71 * dm[(i0+2)*nao+(k0+1)];
                    val += gout80 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout27 * dm[(j0+0)*nao+(l0+1)];
                    val += gout54 * dm[(j0+0)*nao+(l0+2)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    val += gout30 * dm[(j0+1)*nao+(l0+1)];
                    val += gout57 * dm[(j0+1)*nao+(l0+2)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    val += gout33 * dm[(j0+2)*nao+(l0+1)];
                    val += gout60 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    val += gout36 * dm[(j0+0)*nao+(l0+1)];
                    val += gout63 * dm[(j0+0)*nao+(l0+2)];
                    val += gout12 * dm[(j0+1)*nao+(l0+0)];
                    val += gout39 * dm[(j0+1)*nao+(l0+1)];
                    val += gout66 * dm[(j0+1)*nao+(l0+2)];
                    val += gout15 * dm[(j0+2)*nao+(l0+0)];
                    val += gout42 * dm[(j0+2)*nao+(l0+1)];
                    val += gout69 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(l0+0)];
                    val += gout45 * dm[(j0+0)*nao+(l0+1)];
                    val += gout72 * dm[(j0+0)*nao+(l0+2)];
                    val += gout21 * dm[(j0+1)*nao+(l0+0)];
                    val += gout48 * dm[(j0+1)*nao+(l0+1)];
                    val += gout75 * dm[(j0+1)*nao+(l0+2)];
                    val += gout24 * dm[(j0+2)*nao+(l0+0)];
                    val += gout51 * dm[(j0+2)*nao+(l0+1)];
                    val += gout78 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout28 * dm[(j0+0)*nao+(l0+1)];
                    val += gout55 * dm[(j0+0)*nao+(l0+2)];
                    val += gout4 * dm[(j0+1)*nao+(l0+0)];
                    val += gout31 * dm[(j0+1)*nao+(l0+1)];
                    val += gout58 * dm[(j0+1)*nao+(l0+2)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    val += gout34 * dm[(j0+2)*nao+(l0+1)];
                    val += gout61 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    val += gout37 * dm[(j0+0)*nao+(l0+1)];
                    val += gout64 * dm[(j0+0)*nao+(l0+2)];
                    val += gout13 * dm[(j0+1)*nao+(l0+0)];
                    val += gout40 * dm[(j0+1)*nao+(l0+1)];
                    val += gout67 * dm[(j0+1)*nao+(l0+2)];
                    val += gout16 * dm[(j0+2)*nao+(l0+0)];
                    val += gout43 * dm[(j0+2)*nao+(l0+1)];
                    val += gout70 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout19 * dm[(j0+0)*nao+(l0+0)];
                    val += gout46 * dm[(j0+0)*nao+(l0+1)];
                    val += gout73 * dm[(j0+0)*nao+(l0+2)];
                    val += gout22 * dm[(j0+1)*nao+(l0+0)];
                    val += gout49 * dm[(j0+1)*nao+(l0+1)];
                    val += gout76 * dm[(j0+1)*nao+(l0+2)];
                    val += gout25 * dm[(j0+2)*nao+(l0+0)];
                    val += gout52 * dm[(j0+2)*nao+(l0+1)];
                    val += gout79 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout29 * dm[(j0+0)*nao+(l0+1)];
                    val += gout56 * dm[(j0+0)*nao+(l0+2)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    val += gout32 * dm[(j0+1)*nao+(l0+1)];
                    val += gout59 * dm[(j0+1)*nao+(l0+2)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    val += gout35 * dm[(j0+2)*nao+(l0+1)];
                    val += gout62 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+0)];
                    val += gout38 * dm[(j0+0)*nao+(l0+1)];
                    val += gout65 * dm[(j0+0)*nao+(l0+2)];
                    val += gout14 * dm[(j0+1)*nao+(l0+0)];
                    val += gout41 * dm[(j0+1)*nao+(l0+1)];
                    val += gout68 * dm[(j0+1)*nao+(l0+2)];
                    val += gout17 * dm[(j0+2)*nao+(l0+0)];
                    val += gout44 * dm[(j0+2)*nao+(l0+1)];
                    val += gout71 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout20 * dm[(j0+0)*nao+(l0+0)];
                    val += gout47 * dm[(j0+0)*nao+(l0+1)];
                    val += gout74 * dm[(j0+0)*nao+(l0+2)];
                    val += gout23 * dm[(j0+1)*nao+(l0+0)];
                    val += gout50 * dm[(j0+1)*nao+(l0+1)];
                    val += gout77 * dm[(j0+1)*nao+(l0+2)];
                    val += gout26 * dm[(j0+2)*nao+(l0+0)];
                    val += gout53 * dm[(j0+2)*nao+(l0+1)];
                    val += gout80 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout27 * dm[(i0+0)*nao+(l0+1)];
                    val += gout54 * dm[(i0+0)*nao+(l0+2)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout28 * dm[(i0+1)*nao+(l0+1)];
                    val += gout55 * dm[(i0+1)*nao+(l0+2)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout29 * dm[(i0+2)*nao+(l0+1)];
                    val += gout56 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(l0+0)];
                    val += gout36 * dm[(i0+0)*nao+(l0+1)];
                    val += gout63 * dm[(i0+0)*nao+(l0+2)];
                    val += gout10 * dm[(i0+1)*nao+(l0+0)];
                    val += gout37 * dm[(i0+1)*nao+(l0+1)];
                    val += gout64 * dm[(i0+1)*nao+(l0+2)];
                    val += gout11 * dm[(i0+2)*nao+(l0+0)];
                    val += gout38 * dm[(i0+2)*nao+(l0+1)];
                    val += gout65 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(l0+0)];
                    val += gout45 * dm[(i0+0)*nao+(l0+1)];
                    val += gout72 * dm[(i0+0)*nao+(l0+2)];
                    val += gout19 * dm[(i0+1)*nao+(l0+0)];
                    val += gout46 * dm[(i0+1)*nao+(l0+1)];
                    val += gout73 * dm[(i0+1)*nao+(l0+2)];
                    val += gout20 * dm[(i0+2)*nao+(l0+0)];
                    val += gout47 * dm[(i0+2)*nao+(l0+1)];
                    val += gout74 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout30 * dm[(i0+0)*nao+(l0+1)];
                    val += gout57 * dm[(i0+0)*nao+(l0+2)];
                    val += gout4 * dm[(i0+1)*nao+(l0+0)];
                    val += gout31 * dm[(i0+1)*nao+(l0+1)];
                    val += gout58 * dm[(i0+1)*nao+(l0+2)];
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    val += gout32 * dm[(i0+2)*nao+(l0+1)];
                    val += gout59 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout39 * dm[(i0+0)*nao+(l0+1)];
                    val += gout66 * dm[(i0+0)*nao+(l0+2)];
                    val += gout13 * dm[(i0+1)*nao+(l0+0)];
                    val += gout40 * dm[(i0+1)*nao+(l0+1)];
                    val += gout67 * dm[(i0+1)*nao+(l0+2)];
                    val += gout14 * dm[(i0+2)*nao+(l0+0)];
                    val += gout41 * dm[(i0+2)*nao+(l0+1)];
                    val += gout68 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout21 * dm[(i0+0)*nao+(l0+0)];
                    val += gout48 * dm[(i0+0)*nao+(l0+1)];
                    val += gout75 * dm[(i0+0)*nao+(l0+2)];
                    val += gout22 * dm[(i0+1)*nao+(l0+0)];
                    val += gout49 * dm[(i0+1)*nao+(l0+1)];
                    val += gout76 * dm[(i0+1)*nao+(l0+2)];
                    val += gout23 * dm[(i0+2)*nao+(l0+0)];
                    val += gout50 * dm[(i0+2)*nao+(l0+1)];
                    val += gout77 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout33 * dm[(i0+0)*nao+(l0+1)];
                    val += gout60 * dm[(i0+0)*nao+(l0+2)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout34 * dm[(i0+1)*nao+(l0+1)];
                    val += gout61 * dm[(i0+1)*nao+(l0+2)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    val += gout35 * dm[(i0+2)*nao+(l0+1)];
                    val += gout62 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout15 * dm[(i0+0)*nao+(l0+0)];
                    val += gout42 * dm[(i0+0)*nao+(l0+1)];
                    val += gout69 * dm[(i0+0)*nao+(l0+2)];
                    val += gout16 * dm[(i0+1)*nao+(l0+0)];
                    val += gout43 * dm[(i0+1)*nao+(l0+1)];
                    val += gout70 * dm[(i0+1)*nao+(l0+2)];
                    val += gout17 * dm[(i0+2)*nao+(l0+0)];
                    val += gout44 * dm[(i0+2)*nao+(l0+1)];
                    val += gout71 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout24 * dm[(i0+0)*nao+(l0+0)];
                    val += gout51 * dm[(i0+0)*nao+(l0+1)];
                    val += gout78 * dm[(i0+0)*nao+(l0+2)];
                    val += gout25 * dm[(i0+1)*nao+(l0+0)];
                    val += gout52 * dm[(i0+1)*nao+(l0+1)];
                    val += gout79 * dm[(i0+1)*nao+(l0+2)];
                    val += gout26 * dm[(i0+2)*nao+(l0+0)];
                    val += gout53 * dm[(i0+2)*nao+(l0+1)];
                    val += gout80 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_1111(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1111(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        gout0 += trr_20x * 1 * wt;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += trr_10x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += trr_10x * 1 * trr_10z;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        gout3 += 1 * trr_20y * wt;
                        gout4 += 1 * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout5 += 1 * 1 * trr_20z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_jk_2000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        gout0 += trr_21x * 1 * wt;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += trr_11x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += trr_11x * 1 * trr_10z;
                        double trr_01x = cpx * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        gout3 += trr_01x * trr_20y * wt;
                        gout4 += trr_01x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout5 += trr_01x * 1 * trr_20z;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gout6 += trr_20x * trr_01y * wt;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        gout7 += trr_10x * trr_11y * wt;
                        gout8 += trr_10x * trr_01y * trr_10z;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout9 += 1 * trr_21y * wt;
                        gout10 += 1 * trr_11y * trr_10z;
                        gout11 += 1 * trr_01y * trr_20z;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout12 += trr_20x * 1 * trr_01z;
                        gout13 += trr_10x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout14 += trr_10x * 1 * trr_11z;
                        gout15 += 1 * trr_20y * trr_01z;
                        gout16 += 1 * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout17 += 1 * 1 * trr_21z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    val += gout13 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    val += gout14 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+1)];
                    val += gout15 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+1)];
                    val += gout16 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+1)];
                    val += gout17 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+0)*nao+(i0+1)];
                    val += gout8 * dm[(j0+0)*nao+(i0+2)];
                    val += gout9 * dm[(j0+0)*nao+(i0+3)];
                    val += gout10 * dm[(j0+0)*nao+(i0+4)];
                    val += gout11 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+0)];
                    val += gout13 * dm[(j0+0)*nao+(i0+1)];
                    val += gout14 * dm[(j0+0)*nao+(i0+2)];
                    val += gout15 * dm[(j0+0)*nao+(i0+3)];
                    val += gout16 * dm[(j0+0)*nao+(i0+4)];
                    val += gout17 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+0)*nao+(k0+1)];
                    val += gout13 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+1)];
                    val += gout14 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+1)];
                    val += gout15 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+0)*nao+(k0+1)];
                    val += gout16 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    val += gout17 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout6 * dm[(i0+0)*nao+(k0+1)];
                    val += gout12 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout7 * dm[(i0+1)*nao+(k0+1)];
                    val += gout13 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout8 * dm[(i0+2)*nao+(k0+1)];
                    val += gout14 * dm[(i0+2)*nao+(k0+2)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout9 * dm[(i0+3)*nao+(k0+1)];
                    val += gout15 * dm[(i0+3)*nao+(k0+2)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout10 * dm[(i0+4)*nao+(k0+1)];
                    val += gout16 * dm[(i0+4)*nao+(k0+2)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    val += gout11 * dm[(i0+5)*nao+(k0+1)];
                    val += gout17 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout17 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    val += gout9 * dm[(i0+3)*nao+(l0+0)];
                    val += gout10 * dm[(i0+4)*nao+(l0+0)];
                    val += gout11 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+1)*nao+(l0+0)];
                    val += gout14 * dm[(i0+2)*nao+(l0+0)];
                    val += gout15 * dm[(i0+3)*nao+(l0+0)];
                    val += gout16 * dm[(i0+4)*nao+(l0+0)];
                    val += gout17 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_jk_2010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2010(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2011(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double gout18;
    double gout19;
    double gout20;
    double gout21;
    double gout22;
    double gout23;
    double gout24;
    double gout25;
    double gout26;
    double gout27;
    double gout28;
    double gout29;
    double gout30;
    double gout31;
    double gout32;
    double gout33;
    double gout34;
    double gout35;
    double gout36;
    double gout37;
    double gout38;
    double gout39;
    double gout40;
    double gout41;
    double gout42;
    double gout43;
    double gout44;
    double gout45;
    double gout46;
    double gout47;
    double gout48;
    double gout49;
    double gout50;
    double gout51;
    double gout52;
    double gout53;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        gout18 = 0;
        gout19 = 0;
        gout20 = 0;
        gout21 = 0;
        gout22 = 0;
        gout23 = 0;
        gout24 = 0;
        gout25 = 0;
        gout26 = 0;
        gout27 = 0;
        gout28 = 0;
        gout29 = 0;
        gout30 = 0;
        gout31 = 0;
        gout32 = 0;
        gout33 = 0;
        gout34 = 0;
        gout35 = 0;
        gout36 = 0;
        gout37 = 0;
        gout38 = 0;
        gout39 = 0;
        gout40 = 0;
        gout41 = 0;
        gout42 = 0;
        gout43 = 0;
        gout44 = 0;
        gout45 = 0;
        gout46 = 0;
        gout47 = 0;
        gout48 = 0;
        gout49 = 0;
        gout50 = 0;
        gout51 = 0;
        gout52 = 0;
        gout53 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double b01 = .5/akl * (1 - rt_akl);
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double hrr_2011x = trr_22x - xlxk * trr_21x;
                        gout0 += hrr_2011x * 1 * wt;
                        double trr_01x = cpx * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += hrr_1011x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += hrr_1011x * 1 * trr_10z;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        gout3 += hrr_0011x * trr_20y * wt;
                        gout4 += hrr_0011x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout5 += hrr_0011x * 1 * trr_20z;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gout6 += hrr_2001x * trr_01y * wt;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        gout7 += hrr_1001x * trr_11y * wt;
                        gout8 += hrr_1001x * trr_01y * trr_10z;
                        double hrr_0001x = trr_01x - xlxk * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout9 += hrr_0001x * trr_21y * wt;
                        gout10 += hrr_0001x * trr_11y * trr_10z;
                        gout11 += hrr_0001x * trr_01y * trr_20z;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout12 += hrr_2001x * 1 * trr_01z;
                        gout13 += hrr_1001x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout14 += hrr_1001x * 1 * trr_11z;
                        gout15 += hrr_0001x * trr_20y * trr_01z;
                        gout16 += hrr_0001x * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout17 += hrr_0001x * 1 * trr_21z;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        gout18 += trr_21x * hrr_0001y * wt;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gout19 += trr_11x * hrr_1001y * wt;
                        gout20 += trr_11x * hrr_0001y * trr_10z;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        gout21 += trr_01x * hrr_2001y * wt;
                        gout22 += trr_01x * hrr_1001y * trr_10z;
                        gout23 += trr_01x * hrr_0001y * trr_20z;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        gout24 += trr_20x * hrr_0011y * wt;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        gout25 += trr_10x * hrr_1011y * wt;
                        gout26 += trr_10x * hrr_0011y * trr_10z;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        gout27 += 1 * hrr_2011y * wt;
                        gout28 += 1 * hrr_1011y * trr_10z;
                        gout29 += 1 * hrr_0011y * trr_20z;
                        gout30 += trr_20x * hrr_0001y * trr_01z;
                        gout31 += trr_10x * hrr_1001y * trr_01z;
                        gout32 += trr_10x * hrr_0001y * trr_11z;
                        gout33 += 1 * hrr_2001y * trr_01z;
                        gout34 += 1 * hrr_1001y * trr_11z;
                        gout35 += 1 * hrr_0001y * trr_21z;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        gout36 += trr_21x * 1 * hrr_0001z;
                        gout37 += trr_11x * trr_10y * hrr_0001z;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gout38 += trr_11x * 1 * hrr_1001z;
                        gout39 += trr_01x * trr_20y * hrr_0001z;
                        gout40 += trr_01x * trr_10y * hrr_1001z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        gout41 += trr_01x * 1 * hrr_2001z;
                        gout42 += trr_20x * trr_01y * hrr_0001z;
                        gout43 += trr_10x * trr_11y * hrr_0001z;
                        gout44 += trr_10x * trr_01y * hrr_1001z;
                        gout45 += 1 * trr_21y * hrr_0001z;
                        gout46 += 1 * trr_11y * hrr_1001z;
                        gout47 += 1 * trr_01y * hrr_2001z;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        gout48 += trr_20x * 1 * hrr_0011z;
                        gout49 += trr_10x * trr_10y * hrr_0011z;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        gout50 += trr_10x * 1 * hrr_1011z;
                        gout51 += 1 * trr_20y * hrr_0011z;
                        gout52 += 1 * trr_10y * hrr_1011z;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        gout53 += 1 * 1 * hrr_2011z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    val += gout18 * dm[(l0+1)*nao+(k0+0)];
                    val += gout24 * dm[(l0+1)*nao+(k0+1)];
                    val += gout30 * dm[(l0+1)*nao+(k0+2)];
                    val += gout36 * dm[(l0+2)*nao+(k0+0)];
                    val += gout42 * dm[(l0+2)*nao+(k0+1)];
                    val += gout48 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    val += gout13 * dm[(l0+0)*nao+(k0+2)];
                    val += gout19 * dm[(l0+1)*nao+(k0+0)];
                    val += gout25 * dm[(l0+1)*nao+(k0+1)];
                    val += gout31 * dm[(l0+1)*nao+(k0+2)];
                    val += gout37 * dm[(l0+2)*nao+(k0+0)];
                    val += gout43 * dm[(l0+2)*nao+(k0+1)];
                    val += gout49 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    val += gout14 * dm[(l0+0)*nao+(k0+2)];
                    val += gout20 * dm[(l0+1)*nao+(k0+0)];
                    val += gout26 * dm[(l0+1)*nao+(k0+1)];
                    val += gout32 * dm[(l0+1)*nao+(k0+2)];
                    val += gout38 * dm[(l0+2)*nao+(k0+0)];
                    val += gout44 * dm[(l0+2)*nao+(k0+1)];
                    val += gout50 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+1)];
                    val += gout15 * dm[(l0+0)*nao+(k0+2)];
                    val += gout21 * dm[(l0+1)*nao+(k0+0)];
                    val += gout27 * dm[(l0+1)*nao+(k0+1)];
                    val += gout33 * dm[(l0+1)*nao+(k0+2)];
                    val += gout39 * dm[(l0+2)*nao+(k0+0)];
                    val += gout45 * dm[(l0+2)*nao+(k0+1)];
                    val += gout51 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+1)];
                    val += gout16 * dm[(l0+0)*nao+(k0+2)];
                    val += gout22 * dm[(l0+1)*nao+(k0+0)];
                    val += gout28 * dm[(l0+1)*nao+(k0+1)];
                    val += gout34 * dm[(l0+1)*nao+(k0+2)];
                    val += gout40 * dm[(l0+2)*nao+(k0+0)];
                    val += gout46 * dm[(l0+2)*nao+(k0+1)];
                    val += gout52 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+1)];
                    val += gout17 * dm[(l0+0)*nao+(k0+2)];
                    val += gout23 * dm[(l0+1)*nao+(k0+0)];
                    val += gout29 * dm[(l0+1)*nao+(k0+1)];
                    val += gout35 * dm[(l0+1)*nao+(k0+2)];
                    val += gout41 * dm[(l0+2)*nao+(k0+0)];
                    val += gout47 * dm[(l0+2)*nao+(k0+1)];
                    val += gout53 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+0)*nao+(i0+1)];
                    val += gout8 * dm[(j0+0)*nao+(i0+2)];
                    val += gout9 * dm[(j0+0)*nao+(i0+3)];
                    val += gout10 * dm[(j0+0)*nao+(i0+4)];
                    val += gout11 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+0)];
                    val += gout13 * dm[(j0+0)*nao+(i0+1)];
                    val += gout14 * dm[(j0+0)*nao+(i0+2)];
                    val += gout15 * dm[(j0+0)*nao+(i0+3)];
                    val += gout16 * dm[(j0+0)*nao+(i0+4)];
                    val += gout17 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(i0+0)];
                    val += gout19 * dm[(j0+0)*nao+(i0+1)];
                    val += gout20 * dm[(j0+0)*nao+(i0+2)];
                    val += gout21 * dm[(j0+0)*nao+(i0+3)];
                    val += gout22 * dm[(j0+0)*nao+(i0+4)];
                    val += gout23 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout24 * dm[(j0+0)*nao+(i0+0)];
                    val += gout25 * dm[(j0+0)*nao+(i0+1)];
                    val += gout26 * dm[(j0+0)*nao+(i0+2)];
                    val += gout27 * dm[(j0+0)*nao+(i0+3)];
                    val += gout28 * dm[(j0+0)*nao+(i0+4)];
                    val += gout29 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout30 * dm[(j0+0)*nao+(i0+0)];
                    val += gout31 * dm[(j0+0)*nao+(i0+1)];
                    val += gout32 * dm[(j0+0)*nao+(i0+2)];
                    val += gout33 * dm[(j0+0)*nao+(i0+3)];
                    val += gout34 * dm[(j0+0)*nao+(i0+4)];
                    val += gout35 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout36 * dm[(j0+0)*nao+(i0+0)];
                    val += gout37 * dm[(j0+0)*nao+(i0+1)];
                    val += gout38 * dm[(j0+0)*nao+(i0+2)];
                    val += gout39 * dm[(j0+0)*nao+(i0+3)];
                    val += gout40 * dm[(j0+0)*nao+(i0+4)];
                    val += gout41 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout42 * dm[(j0+0)*nao+(i0+0)];
                    val += gout43 * dm[(j0+0)*nao+(i0+1)];
                    val += gout44 * dm[(j0+0)*nao+(i0+2)];
                    val += gout45 * dm[(j0+0)*nao+(i0+3)];
                    val += gout46 * dm[(j0+0)*nao+(i0+4)];
                    val += gout47 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout48 * dm[(j0+0)*nao+(i0+0)];
                    val += gout49 * dm[(j0+0)*nao+(i0+1)];
                    val += gout50 * dm[(j0+0)*nao+(i0+2)];
                    val += gout51 * dm[(j0+0)*nao+(i0+3)];
                    val += gout52 * dm[(j0+0)*nao+(i0+4)];
                    val += gout53 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(k0+0)];
                    val += gout24 * dm[(j0+0)*nao+(k0+1)];
                    val += gout30 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout36 * dm[(j0+0)*nao+(k0+0)];
                    val += gout42 * dm[(j0+0)*nao+(k0+1)];
                    val += gout48 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+0)*nao+(k0+1)];
                    val += gout13 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout19 * dm[(j0+0)*nao+(k0+0)];
                    val += gout25 * dm[(j0+0)*nao+(k0+1)];
                    val += gout31 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout37 * dm[(j0+0)*nao+(k0+0)];
                    val += gout43 * dm[(j0+0)*nao+(k0+1)];
                    val += gout49 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+1)];
                    val += gout14 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout20 * dm[(j0+0)*nao+(k0+0)];
                    val += gout26 * dm[(j0+0)*nao+(k0+1)];
                    val += gout32 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout38 * dm[(j0+0)*nao+(k0+0)];
                    val += gout44 * dm[(j0+0)*nao+(k0+1)];
                    val += gout50 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+1)];
                    val += gout15 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout21 * dm[(j0+0)*nao+(k0+0)];
                    val += gout27 * dm[(j0+0)*nao+(k0+1)];
                    val += gout33 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+1), val);
                    val = 0;
                    val += gout39 * dm[(j0+0)*nao+(k0+0)];
                    val += gout45 * dm[(j0+0)*nao+(k0+1)];
                    val += gout51 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+0)*nao+(k0+1)];
                    val += gout16 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout22 * dm[(j0+0)*nao+(k0+0)];
                    val += gout28 * dm[(j0+0)*nao+(k0+1)];
                    val += gout34 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+1), val);
                    val = 0;
                    val += gout40 * dm[(j0+0)*nao+(k0+0)];
                    val += gout46 * dm[(j0+0)*nao+(k0+1)];
                    val += gout52 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    val += gout17 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout23 * dm[(j0+0)*nao+(k0+0)];
                    val += gout29 * dm[(j0+0)*nao+(k0+1)];
                    val += gout35 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+1), val);
                    val = 0;
                    val += gout41 * dm[(j0+0)*nao+(k0+0)];
                    val += gout47 * dm[(j0+0)*nao+(k0+1)];
                    val += gout53 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout6 * dm[(i0+0)*nao+(k0+1)];
                    val += gout12 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout7 * dm[(i0+1)*nao+(k0+1)];
                    val += gout13 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout8 * dm[(i0+2)*nao+(k0+1)];
                    val += gout14 * dm[(i0+2)*nao+(k0+2)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout9 * dm[(i0+3)*nao+(k0+1)];
                    val += gout15 * dm[(i0+3)*nao+(k0+2)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout10 * dm[(i0+4)*nao+(k0+1)];
                    val += gout16 * dm[(i0+4)*nao+(k0+2)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    val += gout11 * dm[(i0+5)*nao+(k0+1)];
                    val += gout17 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(k0+0)];
                    val += gout24 * dm[(i0+0)*nao+(k0+1)];
                    val += gout30 * dm[(i0+0)*nao+(k0+2)];
                    val += gout19 * dm[(i0+1)*nao+(k0+0)];
                    val += gout25 * dm[(i0+1)*nao+(k0+1)];
                    val += gout31 * dm[(i0+1)*nao+(k0+2)];
                    val += gout20 * dm[(i0+2)*nao+(k0+0)];
                    val += gout26 * dm[(i0+2)*nao+(k0+1)];
                    val += gout32 * dm[(i0+2)*nao+(k0+2)];
                    val += gout21 * dm[(i0+3)*nao+(k0+0)];
                    val += gout27 * dm[(i0+3)*nao+(k0+1)];
                    val += gout33 * dm[(i0+3)*nao+(k0+2)];
                    val += gout22 * dm[(i0+4)*nao+(k0+0)];
                    val += gout28 * dm[(i0+4)*nao+(k0+1)];
                    val += gout34 * dm[(i0+4)*nao+(k0+2)];
                    val += gout23 * dm[(i0+5)*nao+(k0+0)];
                    val += gout29 * dm[(i0+5)*nao+(k0+1)];
                    val += gout35 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout36 * dm[(i0+0)*nao+(k0+0)];
                    val += gout42 * dm[(i0+0)*nao+(k0+1)];
                    val += gout48 * dm[(i0+0)*nao+(k0+2)];
                    val += gout37 * dm[(i0+1)*nao+(k0+0)];
                    val += gout43 * dm[(i0+1)*nao+(k0+1)];
                    val += gout49 * dm[(i0+1)*nao+(k0+2)];
                    val += gout38 * dm[(i0+2)*nao+(k0+0)];
                    val += gout44 * dm[(i0+2)*nao+(k0+1)];
                    val += gout50 * dm[(i0+2)*nao+(k0+2)];
                    val += gout39 * dm[(i0+3)*nao+(k0+0)];
                    val += gout45 * dm[(i0+3)*nao+(k0+1)];
                    val += gout51 * dm[(i0+3)*nao+(k0+2)];
                    val += gout40 * dm[(i0+4)*nao+(k0+0)];
                    val += gout46 * dm[(i0+4)*nao+(k0+1)];
                    val += gout52 * dm[(i0+4)*nao+(k0+2)];
                    val += gout41 * dm[(i0+5)*nao+(k0+0)];
                    val += gout47 * dm[(i0+5)*nao+(k0+1)];
                    val += gout53 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout18 * dm[(j0+0)*nao+(l0+1)];
                    val += gout36 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    val += gout24 * dm[(j0+0)*nao+(l0+1)];
                    val += gout42 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(l0+0)];
                    val += gout30 * dm[(j0+0)*nao+(l0+1)];
                    val += gout48 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout19 * dm[(j0+0)*nao+(l0+1)];
                    val += gout37 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    val += gout25 * dm[(j0+0)*nao+(l0+1)];
                    val += gout43 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(l0+0)];
                    val += gout31 * dm[(j0+0)*nao+(l0+1)];
                    val += gout49 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout20 * dm[(j0+0)*nao+(l0+1)];
                    val += gout38 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    val += gout26 * dm[(j0+0)*nao+(l0+1)];
                    val += gout44 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(l0+0)];
                    val += gout32 * dm[(j0+0)*nao+(l0+1)];
                    val += gout50 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout21 * dm[(j0+0)*nao+(l0+1)];
                    val += gout39 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    val += gout27 * dm[(j0+0)*nao+(l0+1)];
                    val += gout45 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(l0+0)];
                    val += gout33 * dm[(j0+0)*nao+(l0+1)];
                    val += gout51 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout22 * dm[(j0+0)*nao+(l0+1)];
                    val += gout40 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    val += gout28 * dm[(j0+0)*nao+(l0+1)];
                    val += gout46 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(l0+0)];
                    val += gout34 * dm[(j0+0)*nao+(l0+1)];
                    val += gout52 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout23 * dm[(j0+0)*nao+(l0+1)];
                    val += gout41 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+0)];
                    val += gout29 * dm[(j0+0)*nao+(l0+1)];
                    val += gout47 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout17 * dm[(j0+0)*nao+(l0+0)];
                    val += gout35 * dm[(j0+0)*nao+(l0+1)];
                    val += gout53 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout18 * dm[(i0+0)*nao+(l0+1)];
                    val += gout36 * dm[(i0+0)*nao+(l0+2)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout19 * dm[(i0+1)*nao+(l0+1)];
                    val += gout37 * dm[(i0+1)*nao+(l0+2)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout20 * dm[(i0+2)*nao+(l0+1)];
                    val += gout38 * dm[(i0+2)*nao+(l0+2)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout21 * dm[(i0+3)*nao+(l0+1)];
                    val += gout39 * dm[(i0+3)*nao+(l0+2)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout22 * dm[(i0+4)*nao+(l0+1)];
                    val += gout40 * dm[(i0+4)*nao+(l0+2)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    val += gout23 * dm[(i0+5)*nao+(l0+1)];
                    val += gout41 * dm[(i0+5)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout24 * dm[(i0+0)*nao+(l0+1)];
                    val += gout42 * dm[(i0+0)*nao+(l0+2)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout25 * dm[(i0+1)*nao+(l0+1)];
                    val += gout43 * dm[(i0+1)*nao+(l0+2)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    val += gout26 * dm[(i0+2)*nao+(l0+1)];
                    val += gout44 * dm[(i0+2)*nao+(l0+2)];
                    val += gout9 * dm[(i0+3)*nao+(l0+0)];
                    val += gout27 * dm[(i0+3)*nao+(l0+1)];
                    val += gout45 * dm[(i0+3)*nao+(l0+2)];
                    val += gout10 * dm[(i0+4)*nao+(l0+0)];
                    val += gout28 * dm[(i0+4)*nao+(l0+1)];
                    val += gout46 * dm[(i0+4)*nao+(l0+2)];
                    val += gout11 * dm[(i0+5)*nao+(l0+0)];
                    val += gout29 * dm[(i0+5)*nao+(l0+1)];
                    val += gout47 * dm[(i0+5)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout30 * dm[(i0+0)*nao+(l0+1)];
                    val += gout48 * dm[(i0+0)*nao+(l0+2)];
                    val += gout13 * dm[(i0+1)*nao+(l0+0)];
                    val += gout31 * dm[(i0+1)*nao+(l0+1)];
                    val += gout49 * dm[(i0+1)*nao+(l0+2)];
                    val += gout14 * dm[(i0+2)*nao+(l0+0)];
                    val += gout32 * dm[(i0+2)*nao+(l0+1)];
                    val += gout50 * dm[(i0+2)*nao+(l0+2)];
                    val += gout15 * dm[(i0+3)*nao+(l0+0)];
                    val += gout33 * dm[(i0+3)*nao+(l0+1)];
                    val += gout51 * dm[(i0+3)*nao+(l0+2)];
                    val += gout16 * dm[(i0+4)*nao+(l0+0)];
                    val += gout34 * dm[(i0+4)*nao+(l0+1)];
                    val += gout52 * dm[(i0+4)*nao+(l0+2)];
                    val += gout17 * dm[(i0+5)*nao+(l0+0)];
                    val += gout35 * dm[(i0+5)*nao+(l0+1)];
                    val += gout53 * dm[(i0+5)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2011(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2011(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2020(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double gout18;
    double gout19;
    double gout20;
    double gout21;
    double gout22;
    double gout23;
    double gout24;
    double gout25;
    double gout26;
    double gout27;
    double gout28;
    double gout29;
    double gout30;
    double gout31;
    double gout32;
    double gout33;
    double gout34;
    double gout35;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        gout18 = 0;
        gout19 = 0;
        gout20 = 0;
        gout21 = 0;
        gout22 = 0;
        gout23 = 0;
        gout24 = 0;
        gout25 = 0;
        gout26 = 0;
        gout27 = 0;
        gout28 = 0;
        gout29 = 0;
        gout30 = 0;
        gout31 = 0;
        gout32 = 0;
        gout33 = 0;
        gout34 = 0;
        gout35 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double b01 = .5/akl * (1 - rt_akl);
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        gout0 += trr_22x * 1 * wt;
                        double trr_01x = cpx * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += trr_12x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += trr_12x * 1 * trr_10z;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        gout3 += trr_02x * trr_20y * wt;
                        gout4 += trr_02x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout5 += trr_02x * 1 * trr_20z;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gout6 += trr_21x * trr_01y * wt;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        gout7 += trr_11x * trr_11y * wt;
                        gout8 += trr_11x * trr_01y * trr_10z;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout9 += trr_01x * trr_21y * wt;
                        gout10 += trr_01x * trr_11y * trr_10z;
                        gout11 += trr_01x * trr_01y * trr_20z;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout12 += trr_21x * 1 * trr_01z;
                        gout13 += trr_11x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout14 += trr_11x * 1 * trr_11z;
                        gout15 += trr_01x * trr_20y * trr_01z;
                        gout16 += trr_01x * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout17 += trr_01x * 1 * trr_21z;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        gout18 += trr_20x * trr_02y * wt;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        gout19 += trr_10x * trr_12y * wt;
                        gout20 += trr_10x * trr_02y * trr_10z;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        gout21 += 1 * trr_22y * wt;
                        gout22 += 1 * trr_12y * trr_10z;
                        gout23 += 1 * trr_02y * trr_20z;
                        gout24 += trr_20x * trr_01y * trr_01z;
                        gout25 += trr_10x * trr_11y * trr_01z;
                        gout26 += trr_10x * trr_01y * trr_11z;
                        gout27 += 1 * trr_21y * trr_01z;
                        gout28 += 1 * trr_11y * trr_11z;
                        gout29 += 1 * trr_01y * trr_21z;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        gout30 += trr_20x * 1 * trr_02z;
                        gout31 += trr_10x * trr_10y * trr_02z;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        gout32 += trr_10x * 1 * trr_12z;
                        gout33 += 1 * trr_20y * trr_02z;
                        gout34 += 1 * trr_10y * trr_12z;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        gout35 += 1 * 1 * trr_22z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    val += gout18 * dm[(l0+0)*nao+(k0+3)];
                    val += gout24 * dm[(l0+0)*nao+(k0+4)];
                    val += gout30 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    val += gout13 * dm[(l0+0)*nao+(k0+2)];
                    val += gout19 * dm[(l0+0)*nao+(k0+3)];
                    val += gout25 * dm[(l0+0)*nao+(k0+4)];
                    val += gout31 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    val += gout14 * dm[(l0+0)*nao+(k0+2)];
                    val += gout20 * dm[(l0+0)*nao+(k0+3)];
                    val += gout26 * dm[(l0+0)*nao+(k0+4)];
                    val += gout32 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+1)];
                    val += gout15 * dm[(l0+0)*nao+(k0+2)];
                    val += gout21 * dm[(l0+0)*nao+(k0+3)];
                    val += gout27 * dm[(l0+0)*nao+(k0+4)];
                    val += gout33 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+1)];
                    val += gout16 * dm[(l0+0)*nao+(k0+2)];
                    val += gout22 * dm[(l0+0)*nao+(k0+3)];
                    val += gout28 * dm[(l0+0)*nao+(k0+4)];
                    val += gout34 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+1)];
                    val += gout17 * dm[(l0+0)*nao+(k0+2)];
                    val += gout23 * dm[(l0+0)*nao+(k0+3)];
                    val += gout29 * dm[(l0+0)*nao+(k0+4)];
                    val += gout35 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+0)*nao+(i0+1)];
                    val += gout8 * dm[(j0+0)*nao+(i0+2)];
                    val += gout9 * dm[(j0+0)*nao+(i0+3)];
                    val += gout10 * dm[(j0+0)*nao+(i0+4)];
                    val += gout11 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+0)];
                    val += gout13 * dm[(j0+0)*nao+(i0+1)];
                    val += gout14 * dm[(j0+0)*nao+(i0+2)];
                    val += gout15 * dm[(j0+0)*nao+(i0+3)];
                    val += gout16 * dm[(j0+0)*nao+(i0+4)];
                    val += gout17 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(i0+0)];
                    val += gout19 * dm[(j0+0)*nao+(i0+1)];
                    val += gout20 * dm[(j0+0)*nao+(i0+2)];
                    val += gout21 * dm[(j0+0)*nao+(i0+3)];
                    val += gout22 * dm[(j0+0)*nao+(i0+4)];
                    val += gout23 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout24 * dm[(j0+0)*nao+(i0+0)];
                    val += gout25 * dm[(j0+0)*nao+(i0+1)];
                    val += gout26 * dm[(j0+0)*nao+(i0+2)];
                    val += gout27 * dm[(j0+0)*nao+(i0+3)];
                    val += gout28 * dm[(j0+0)*nao+(i0+4)];
                    val += gout29 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout30 * dm[(j0+0)*nao+(i0+0)];
                    val += gout31 * dm[(j0+0)*nao+(i0+1)];
                    val += gout32 * dm[(j0+0)*nao+(i0+2)];
                    val += gout33 * dm[(j0+0)*nao+(i0+3)];
                    val += gout34 * dm[(j0+0)*nao+(i0+4)];
                    val += gout35 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+5)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    val += gout18 * dm[(j0+0)*nao+(k0+3)];
                    val += gout24 * dm[(j0+0)*nao+(k0+4)];
                    val += gout30 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+0)*nao+(k0+1)];
                    val += gout13 * dm[(j0+0)*nao+(k0+2)];
                    val += gout19 * dm[(j0+0)*nao+(k0+3)];
                    val += gout25 * dm[(j0+0)*nao+(k0+4)];
                    val += gout31 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+1)];
                    val += gout14 * dm[(j0+0)*nao+(k0+2)];
                    val += gout20 * dm[(j0+0)*nao+(k0+3)];
                    val += gout26 * dm[(j0+0)*nao+(k0+4)];
                    val += gout32 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+1)];
                    val += gout15 * dm[(j0+0)*nao+(k0+2)];
                    val += gout21 * dm[(j0+0)*nao+(k0+3)];
                    val += gout27 * dm[(j0+0)*nao+(k0+4)];
                    val += gout33 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+0)*nao+(k0+1)];
                    val += gout16 * dm[(j0+0)*nao+(k0+2)];
                    val += gout22 * dm[(j0+0)*nao+(k0+3)];
                    val += gout28 * dm[(j0+0)*nao+(k0+4)];
                    val += gout34 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    val += gout17 * dm[(j0+0)*nao+(k0+2)];
                    val += gout23 * dm[(j0+0)*nao+(k0+3)];
                    val += gout29 * dm[(j0+0)*nao+(k0+4)];
                    val += gout35 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout6 * dm[(i0+0)*nao+(k0+1)];
                    val += gout12 * dm[(i0+0)*nao+(k0+2)];
                    val += gout18 * dm[(i0+0)*nao+(k0+3)];
                    val += gout24 * dm[(i0+0)*nao+(k0+4)];
                    val += gout30 * dm[(i0+0)*nao+(k0+5)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout7 * dm[(i0+1)*nao+(k0+1)];
                    val += gout13 * dm[(i0+1)*nao+(k0+2)];
                    val += gout19 * dm[(i0+1)*nao+(k0+3)];
                    val += gout25 * dm[(i0+1)*nao+(k0+4)];
                    val += gout31 * dm[(i0+1)*nao+(k0+5)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout8 * dm[(i0+2)*nao+(k0+1)];
                    val += gout14 * dm[(i0+2)*nao+(k0+2)];
                    val += gout20 * dm[(i0+2)*nao+(k0+3)];
                    val += gout26 * dm[(i0+2)*nao+(k0+4)];
                    val += gout32 * dm[(i0+2)*nao+(k0+5)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout9 * dm[(i0+3)*nao+(k0+1)];
                    val += gout15 * dm[(i0+3)*nao+(k0+2)];
                    val += gout21 * dm[(i0+3)*nao+(k0+3)];
                    val += gout27 * dm[(i0+3)*nao+(k0+4)];
                    val += gout33 * dm[(i0+3)*nao+(k0+5)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout10 * dm[(i0+4)*nao+(k0+1)];
                    val += gout16 * dm[(i0+4)*nao+(k0+2)];
                    val += gout22 * dm[(i0+4)*nao+(k0+3)];
                    val += gout28 * dm[(i0+4)*nao+(k0+4)];
                    val += gout34 * dm[(i0+4)*nao+(k0+5)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    val += gout11 * dm[(i0+5)*nao+(k0+1)];
                    val += gout17 * dm[(i0+5)*nao+(k0+2)];
                    val += gout23 * dm[(i0+5)*nao+(k0+3)];
                    val += gout29 * dm[(i0+5)*nao+(k0+4)];
                    val += gout35 * dm[(i0+5)*nao+(k0+5)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+3), val);
                    val = 0;
                    val += gout24 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+4), val);
                    val = 0;
                    val += gout30 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+5), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout19 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+3), val);
                    val = 0;
                    val += gout25 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+4), val);
                    val = 0;
                    val += gout31 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+5), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout20 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+3), val);
                    val = 0;
                    val += gout26 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+4), val);
                    val = 0;
                    val += gout32 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+5), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout21 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+3), val);
                    val = 0;
                    val += gout27 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+4), val);
                    val = 0;
                    val += gout33 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+5), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout22 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+3), val);
                    val = 0;
                    val += gout28 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+4), val);
                    val = 0;
                    val += gout34 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+5), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout17 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout23 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+3), val);
                    val = 0;
                    val += gout29 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+4), val);
                    val = 0;
                    val += gout35 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+5), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    val += gout9 * dm[(i0+3)*nao+(l0+0)];
                    val += gout10 * dm[(i0+4)*nao+(l0+0)];
                    val += gout11 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+1)*nao+(l0+0)];
                    val += gout14 * dm[(i0+2)*nao+(l0+0)];
                    val += gout15 * dm[(i0+3)*nao+(l0+0)];
                    val += gout16 * dm[(i0+4)*nao+(l0+0)];
                    val += gout17 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(l0+0)];
                    val += gout19 * dm[(i0+1)*nao+(l0+0)];
                    val += gout20 * dm[(i0+2)*nao+(l0+0)];
                    val += gout21 * dm[(i0+3)*nao+(l0+0)];
                    val += gout22 * dm[(i0+4)*nao+(l0+0)];
                    val += gout23 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+3), val);
                    val = 0;
                    val += gout24 * dm[(i0+0)*nao+(l0+0)];
                    val += gout25 * dm[(i0+1)*nao+(l0+0)];
                    val += gout26 * dm[(i0+2)*nao+(l0+0)];
                    val += gout27 * dm[(i0+3)*nao+(l0+0)];
                    val += gout28 * dm[(i0+4)*nao+(l0+0)];
                    val += gout29 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+4), val);
                    val = 0;
                    val += gout30 * dm[(i0+0)*nao+(l0+0)];
                    val += gout31 * dm[(i0+1)*nao+(l0+0)];
                    val += gout32 * dm[(i0+2)*nao+(l0+0)];
                    val += gout33 * dm[(i0+3)*nao+(l0+0)];
                    val += gout34 * dm[(i0+4)*nao+(l0+0)];
                    val += gout35 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+5), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2020(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2020(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        gout0 += hrr_2100x * 1 * wt;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += hrr_1100x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += hrr_1100x * 1 * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        gout3 += hrr_0100x * trr_20y * wt;
                        gout4 += hrr_0100x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout5 += hrr_0100x * 1 * trr_20z;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gout6 += trr_20x * hrr_0100y * wt;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout7 += trr_10x * hrr_1100y * wt;
                        gout8 += trr_10x * hrr_0100y * trr_10z;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        gout9 += 1 * hrr_2100y * wt;
                        gout10 += 1 * hrr_1100y * trr_10z;
                        gout11 += 1 * hrr_0100y * trr_20z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout12 += trr_20x * 1 * hrr_0100z;
                        gout13 += trr_10x * trr_10y * hrr_0100z;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout14 += trr_10x * 1 * hrr_1100z;
                        gout15 += 1 * trr_20y * hrr_0100z;
                        gout16 += 1 * trr_10y * hrr_1100z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        gout17 += 1 * 1 * hrr_2100z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout12 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout13 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout14 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+1), val);
                    val = 0;
                    val += gout15 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+2), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout10 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+1), val);
                    val = 0;
                    val += gout16 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+2), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout11 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+1), val);
                    val = 0;
                    val += gout17 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    val += gout6 * dm[(j0+1)*nao+(i0+0)];
                    val += gout7 * dm[(j0+1)*nao+(i0+1)];
                    val += gout8 * dm[(j0+1)*nao+(i0+2)];
                    val += gout9 * dm[(j0+1)*nao+(i0+3)];
                    val += gout10 * dm[(j0+1)*nao+(i0+4)];
                    val += gout11 * dm[(j0+1)*nao+(i0+5)];
                    val += gout12 * dm[(j0+2)*nao+(i0+0)];
                    val += gout13 * dm[(j0+2)*nao+(i0+1)];
                    val += gout14 * dm[(j0+2)*nao+(i0+2)];
                    val += gout15 * dm[(j0+2)*nao+(i0+3)];
                    val += gout16 * dm[(j0+2)*nao+(i0+4)];
                    val += gout17 * dm[(j0+2)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout6 * dm[(j0+1)*nao+(k0+0)];
                    val += gout12 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+1)*nao+(k0+0)];
                    val += gout13 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+1)*nao+(k0+0)];
                    val += gout14 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+1)*nao+(k0+0)];
                    val += gout15 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+1)*nao+(k0+0)];
                    val += gout16 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout11 * dm[(j0+1)*nao+(k0+0)];
                    val += gout17 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+0)];
                    val += gout7 * dm[(i0+1)*nao+(k0+0)];
                    val += gout8 * dm[(i0+2)*nao+(k0+0)];
                    val += gout9 * dm[(i0+3)*nao+(k0+0)];
                    val += gout10 * dm[(i0+4)*nao+(k0+0)];
                    val += gout11 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(k0+0)];
                    val += gout13 * dm[(i0+1)*nao+(k0+0)];
                    val += gout14 * dm[(i0+2)*nao+(k0+0)];
                    val += gout15 * dm[(i0+3)*nao+(k0+0)];
                    val += gout16 * dm[(i0+4)*nao+(k0+0)];
                    val += gout17 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout7 * dm[(j0+1)*nao+(l0+0)];
                    val += gout13 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+1)*nao+(l0+0)];
                    val += gout14 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+1)*nao+(l0+0)];
                    val += gout15 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+1)*nao+(l0+0)];
                    val += gout16 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+1)*nao+(l0+0)];
                    val += gout17 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    val += gout9 * dm[(i0+3)*nao+(l0+0)];
                    val += gout10 * dm[(i0+4)*nao+(l0+0)];
                    val += gout11 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+1)*nao+(l0+0)];
                    val += gout14 * dm[(i0+2)*nao+(l0+0)];
                    val += gout15 * dm[(i0+3)*nao+(l0+0)];
                    val += gout16 * dm[(i0+4)*nao+(l0+0)];
                    val += gout17 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_jk_2100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2100(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2110(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double gout18;
    double gout19;
    double gout20;
    double gout21;
    double gout22;
    double gout23;
    double gout24;
    double gout25;
    double gout26;
    double gout27;
    double gout28;
    double gout29;
    double gout30;
    double gout31;
    double gout32;
    double gout33;
    double gout34;
    double gout35;
    double gout36;
    double gout37;
    double gout38;
    double gout39;
    double gout40;
    double gout41;
    double gout42;
    double gout43;
    double gout44;
    double gout45;
    double gout46;
    double gout47;
    double gout48;
    double gout49;
    double gout50;
    double gout51;
    double gout52;
    double gout53;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        gout18 = 0;
        gout19 = 0;
        gout20 = 0;
        gout21 = 0;
        gout22 = 0;
        gout23 = 0;
        gout24 = 0;
        gout25 = 0;
        gout26 = 0;
        gout27 = 0;
        gout28 = 0;
        gout29 = 0;
        gout30 = 0;
        gout31 = 0;
        gout32 = 0;
        gout33 = 0;
        gout34 = 0;
        gout35 = 0;
        gout36 = 0;
        gout37 = 0;
        gout38 = 0;
        gout39 = 0;
        gout40 = 0;
        gout41 = 0;
        gout42 = 0;
        gout43 = 0;
        gout44 = 0;
        gout45 = 0;
        gout46 = 0;
        gout47 = 0;
        gout48 = 0;
        gout49 = 0;
        gout50 = 0;
        gout51 = 0;
        gout52 = 0;
        gout53 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double hrr_2110x = trr_31x - xjxi * trr_21x;
                        gout0 += hrr_2110x * 1 * wt;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += hrr_1110x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += hrr_1110x * 1 * trr_10z;
                        double trr_01x = cpx * 1;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        gout3 += hrr_0110x * trr_20y * wt;
                        gout4 += hrr_0110x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout5 += hrr_0110x * 1 * trr_20z;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gout6 += trr_21x * hrr_0100y * wt;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout7 += trr_11x * hrr_1100y * wt;
                        gout8 += trr_11x * hrr_0100y * trr_10z;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        gout9 += trr_01x * hrr_2100y * wt;
                        gout10 += trr_01x * hrr_1100y * trr_10z;
                        gout11 += trr_01x * hrr_0100y * trr_20z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout12 += trr_21x * 1 * hrr_0100z;
                        gout13 += trr_11x * trr_10y * hrr_0100z;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout14 += trr_11x * 1 * hrr_1100z;
                        gout15 += trr_01x * trr_20y * hrr_0100z;
                        gout16 += trr_01x * trr_10y * hrr_1100z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        gout17 += trr_01x * 1 * hrr_2100z;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gout18 += hrr_2100x * trr_01y * wt;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        gout19 += hrr_1100x * trr_11y * wt;
                        gout20 += hrr_1100x * trr_01y * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout21 += hrr_0100x * trr_21y * wt;
                        gout22 += hrr_0100x * trr_11y * trr_10z;
                        gout23 += hrr_0100x * trr_01y * trr_20z;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gout24 += trr_20x * hrr_0110y * wt;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gout25 += trr_10x * hrr_1110y * wt;
                        gout26 += trr_10x * hrr_0110y * trr_10z;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double hrr_2110y = trr_31y - yjyi * trr_21y;
                        gout27 += 1 * hrr_2110y * wt;
                        gout28 += 1 * hrr_1110y * trr_10z;
                        gout29 += 1 * hrr_0110y * trr_20z;
                        gout30 += trr_20x * trr_01y * hrr_0100z;
                        gout31 += trr_10x * trr_11y * hrr_0100z;
                        gout32 += trr_10x * trr_01y * hrr_1100z;
                        gout33 += 1 * trr_21y * hrr_0100z;
                        gout34 += 1 * trr_11y * hrr_1100z;
                        gout35 += 1 * trr_01y * hrr_2100z;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout36 += hrr_2100x * 1 * trr_01z;
                        gout37 += hrr_1100x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout38 += hrr_1100x * 1 * trr_11z;
                        gout39 += hrr_0100x * trr_20y * trr_01z;
                        gout40 += hrr_0100x * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout41 += hrr_0100x * 1 * trr_21z;
                        gout42 += trr_20x * hrr_0100y * trr_01z;
                        gout43 += trr_10x * hrr_1100y * trr_01z;
                        gout44 += trr_10x * hrr_0100y * trr_11z;
                        gout45 += 1 * hrr_2100y * trr_01z;
                        gout46 += 1 * hrr_1100y * trr_11z;
                        gout47 += 1 * hrr_0100y * trr_21z;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gout48 += trr_20x * 1 * hrr_0110z;
                        gout49 += trr_10x * trr_10y * hrr_0110z;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gout50 += trr_10x * 1 * hrr_1110z;
                        gout51 += 1 * trr_20y * hrr_0110z;
                        gout52 += 1 * trr_10y * hrr_1110z;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double hrr_2110z = trr_31z - zjzi * trr_21z;
                        gout53 += 1 * 1 * hrr_2110z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout18 * dm[(l0+0)*nao+(k0+1)];
                    val += gout36 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    val += gout24 * dm[(l0+0)*nao+(k0+1)];
                    val += gout42 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout12 * dm[(l0+0)*nao+(k0+0)];
                    val += gout30 * dm[(l0+0)*nao+(k0+1)];
                    val += gout48 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout19 * dm[(l0+0)*nao+(k0+1)];
                    val += gout37 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    val += gout25 * dm[(l0+0)*nao+(k0+1)];
                    val += gout43 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout13 * dm[(l0+0)*nao+(k0+0)];
                    val += gout31 * dm[(l0+0)*nao+(k0+1)];
                    val += gout49 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout20 * dm[(l0+0)*nao+(k0+1)];
                    val += gout38 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    val += gout26 * dm[(l0+0)*nao+(k0+1)];
                    val += gout44 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout14 * dm[(l0+0)*nao+(k0+0)];
                    val += gout32 * dm[(l0+0)*nao+(k0+1)];
                    val += gout50 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout21 * dm[(l0+0)*nao+(k0+1)];
                    val += gout39 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    val += gout27 * dm[(l0+0)*nao+(k0+1)];
                    val += gout45 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+1), val);
                    val = 0;
                    val += gout15 * dm[(l0+0)*nao+(k0+0)];
                    val += gout33 * dm[(l0+0)*nao+(k0+1)];
                    val += gout51 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+2), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout22 * dm[(l0+0)*nao+(k0+1)];
                    val += gout40 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout10 * dm[(l0+0)*nao+(k0+0)];
                    val += gout28 * dm[(l0+0)*nao+(k0+1)];
                    val += gout46 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+1), val);
                    val = 0;
                    val += gout16 * dm[(l0+0)*nao+(k0+0)];
                    val += gout34 * dm[(l0+0)*nao+(k0+1)];
                    val += gout52 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+2), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    val += gout23 * dm[(l0+0)*nao+(k0+1)];
                    val += gout41 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout11 * dm[(l0+0)*nao+(k0+0)];
                    val += gout29 * dm[(l0+0)*nao+(k0+1)];
                    val += gout47 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+1), val);
                    val = 0;
                    val += gout17 * dm[(l0+0)*nao+(k0+0)];
                    val += gout35 * dm[(l0+0)*nao+(k0+1)];
                    val += gout53 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    val += gout6 * dm[(j0+1)*nao+(i0+0)];
                    val += gout7 * dm[(j0+1)*nao+(i0+1)];
                    val += gout8 * dm[(j0+1)*nao+(i0+2)];
                    val += gout9 * dm[(j0+1)*nao+(i0+3)];
                    val += gout10 * dm[(j0+1)*nao+(i0+4)];
                    val += gout11 * dm[(j0+1)*nao+(i0+5)];
                    val += gout12 * dm[(j0+2)*nao+(i0+0)];
                    val += gout13 * dm[(j0+2)*nao+(i0+1)];
                    val += gout14 * dm[(j0+2)*nao+(i0+2)];
                    val += gout15 * dm[(j0+2)*nao+(i0+3)];
                    val += gout16 * dm[(j0+2)*nao+(i0+4)];
                    val += gout17 * dm[(j0+2)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(i0+0)];
                    val += gout19 * dm[(j0+0)*nao+(i0+1)];
                    val += gout20 * dm[(j0+0)*nao+(i0+2)];
                    val += gout21 * dm[(j0+0)*nao+(i0+3)];
                    val += gout22 * dm[(j0+0)*nao+(i0+4)];
                    val += gout23 * dm[(j0+0)*nao+(i0+5)];
                    val += gout24 * dm[(j0+1)*nao+(i0+0)];
                    val += gout25 * dm[(j0+1)*nao+(i0+1)];
                    val += gout26 * dm[(j0+1)*nao+(i0+2)];
                    val += gout27 * dm[(j0+1)*nao+(i0+3)];
                    val += gout28 * dm[(j0+1)*nao+(i0+4)];
                    val += gout29 * dm[(j0+1)*nao+(i0+5)];
                    val += gout30 * dm[(j0+2)*nao+(i0+0)];
                    val += gout31 * dm[(j0+2)*nao+(i0+1)];
                    val += gout32 * dm[(j0+2)*nao+(i0+2)];
                    val += gout33 * dm[(j0+2)*nao+(i0+3)];
                    val += gout34 * dm[(j0+2)*nao+(i0+4)];
                    val += gout35 * dm[(j0+2)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout36 * dm[(j0+0)*nao+(i0+0)];
                    val += gout37 * dm[(j0+0)*nao+(i0+1)];
                    val += gout38 * dm[(j0+0)*nao+(i0+2)];
                    val += gout39 * dm[(j0+0)*nao+(i0+3)];
                    val += gout40 * dm[(j0+0)*nao+(i0+4)];
                    val += gout41 * dm[(j0+0)*nao+(i0+5)];
                    val += gout42 * dm[(j0+1)*nao+(i0+0)];
                    val += gout43 * dm[(j0+1)*nao+(i0+1)];
                    val += gout44 * dm[(j0+1)*nao+(i0+2)];
                    val += gout45 * dm[(j0+1)*nao+(i0+3)];
                    val += gout46 * dm[(j0+1)*nao+(i0+4)];
                    val += gout47 * dm[(j0+1)*nao+(i0+5)];
                    val += gout48 * dm[(j0+2)*nao+(i0+0)];
                    val += gout49 * dm[(j0+2)*nao+(i0+1)];
                    val += gout50 * dm[(j0+2)*nao+(i0+2)];
                    val += gout51 * dm[(j0+2)*nao+(i0+3)];
                    val += gout52 * dm[(j0+2)*nao+(i0+4)];
                    val += gout53 * dm[(j0+2)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout18 * dm[(j0+0)*nao+(k0+1)];
                    val += gout36 * dm[(j0+0)*nao+(k0+2)];
                    val += gout6 * dm[(j0+1)*nao+(k0+0)];
                    val += gout24 * dm[(j0+1)*nao+(k0+1)];
                    val += gout42 * dm[(j0+1)*nao+(k0+2)];
                    val += gout12 * dm[(j0+2)*nao+(k0+0)];
                    val += gout30 * dm[(j0+2)*nao+(k0+1)];
                    val += gout48 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout19 * dm[(j0+0)*nao+(k0+1)];
                    val += gout37 * dm[(j0+0)*nao+(k0+2)];
                    val += gout7 * dm[(j0+1)*nao+(k0+0)];
                    val += gout25 * dm[(j0+1)*nao+(k0+1)];
                    val += gout43 * dm[(j0+1)*nao+(k0+2)];
                    val += gout13 * dm[(j0+2)*nao+(k0+0)];
                    val += gout31 * dm[(j0+2)*nao+(k0+1)];
                    val += gout49 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout20 * dm[(j0+0)*nao+(k0+1)];
                    val += gout38 * dm[(j0+0)*nao+(k0+2)];
                    val += gout8 * dm[(j0+1)*nao+(k0+0)];
                    val += gout26 * dm[(j0+1)*nao+(k0+1)];
                    val += gout44 * dm[(j0+1)*nao+(k0+2)];
                    val += gout14 * dm[(j0+2)*nao+(k0+0)];
                    val += gout32 * dm[(j0+2)*nao+(k0+1)];
                    val += gout50 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout21 * dm[(j0+0)*nao+(k0+1)];
                    val += gout39 * dm[(j0+0)*nao+(k0+2)];
                    val += gout9 * dm[(j0+1)*nao+(k0+0)];
                    val += gout27 * dm[(j0+1)*nao+(k0+1)];
                    val += gout45 * dm[(j0+1)*nao+(k0+2)];
                    val += gout15 * dm[(j0+2)*nao+(k0+0)];
                    val += gout33 * dm[(j0+2)*nao+(k0+1)];
                    val += gout51 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout22 * dm[(j0+0)*nao+(k0+1)];
                    val += gout40 * dm[(j0+0)*nao+(k0+2)];
                    val += gout10 * dm[(j0+1)*nao+(k0+0)];
                    val += gout28 * dm[(j0+1)*nao+(k0+1)];
                    val += gout46 * dm[(j0+1)*nao+(k0+2)];
                    val += gout16 * dm[(j0+2)*nao+(k0+0)];
                    val += gout34 * dm[(j0+2)*nao+(k0+1)];
                    val += gout52 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout23 * dm[(j0+0)*nao+(k0+1)];
                    val += gout41 * dm[(j0+0)*nao+(k0+2)];
                    val += gout11 * dm[(j0+1)*nao+(k0+0)];
                    val += gout29 * dm[(j0+1)*nao+(k0+1)];
                    val += gout47 * dm[(j0+1)*nao+(k0+2)];
                    val += gout17 * dm[(j0+2)*nao+(k0+0)];
                    val += gout35 * dm[(j0+2)*nao+(k0+1)];
                    val += gout53 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout18 * dm[(i0+0)*nao+(k0+1)];
                    val += gout36 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout19 * dm[(i0+1)*nao+(k0+1)];
                    val += gout37 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout20 * dm[(i0+2)*nao+(k0+1)];
                    val += gout38 * dm[(i0+2)*nao+(k0+2)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout21 * dm[(i0+3)*nao+(k0+1)];
                    val += gout39 * dm[(i0+3)*nao+(k0+2)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout22 * dm[(i0+4)*nao+(k0+1)];
                    val += gout40 * dm[(i0+4)*nao+(k0+2)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    val += gout23 * dm[(i0+5)*nao+(k0+1)];
                    val += gout41 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+0)];
                    val += gout24 * dm[(i0+0)*nao+(k0+1)];
                    val += gout42 * dm[(i0+0)*nao+(k0+2)];
                    val += gout7 * dm[(i0+1)*nao+(k0+0)];
                    val += gout25 * dm[(i0+1)*nao+(k0+1)];
                    val += gout43 * dm[(i0+1)*nao+(k0+2)];
                    val += gout8 * dm[(i0+2)*nao+(k0+0)];
                    val += gout26 * dm[(i0+2)*nao+(k0+1)];
                    val += gout44 * dm[(i0+2)*nao+(k0+2)];
                    val += gout9 * dm[(i0+3)*nao+(k0+0)];
                    val += gout27 * dm[(i0+3)*nao+(k0+1)];
                    val += gout45 * dm[(i0+3)*nao+(k0+2)];
                    val += gout10 * dm[(i0+4)*nao+(k0+0)];
                    val += gout28 * dm[(i0+4)*nao+(k0+1)];
                    val += gout46 * dm[(i0+4)*nao+(k0+2)];
                    val += gout11 * dm[(i0+5)*nao+(k0+0)];
                    val += gout29 * dm[(i0+5)*nao+(k0+1)];
                    val += gout47 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(k0+0)];
                    val += gout30 * dm[(i0+0)*nao+(k0+1)];
                    val += gout48 * dm[(i0+0)*nao+(k0+2)];
                    val += gout13 * dm[(i0+1)*nao+(k0+0)];
                    val += gout31 * dm[(i0+1)*nao+(k0+1)];
                    val += gout49 * dm[(i0+1)*nao+(k0+2)];
                    val += gout14 * dm[(i0+2)*nao+(k0+0)];
                    val += gout32 * dm[(i0+2)*nao+(k0+1)];
                    val += gout50 * dm[(i0+2)*nao+(k0+2)];
                    val += gout15 * dm[(i0+3)*nao+(k0+0)];
                    val += gout33 * dm[(i0+3)*nao+(k0+1)];
                    val += gout51 * dm[(i0+3)*nao+(k0+2)];
                    val += gout16 * dm[(i0+4)*nao+(k0+0)];
                    val += gout34 * dm[(i0+4)*nao+(k0+1)];
                    val += gout52 * dm[(i0+4)*nao+(k0+2)];
                    val += gout17 * dm[(i0+5)*nao+(k0+0)];
                    val += gout35 * dm[(i0+5)*nao+(k0+1)];
                    val += gout53 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(l0+0)];
                    val += gout24 * dm[(j0+1)*nao+(l0+0)];
                    val += gout30 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout36 * dm[(j0+0)*nao+(l0+0)];
                    val += gout42 * dm[(j0+1)*nao+(l0+0)];
                    val += gout48 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout7 * dm[(j0+1)*nao+(l0+0)];
                    val += gout13 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout19 * dm[(j0+0)*nao+(l0+0)];
                    val += gout25 * dm[(j0+1)*nao+(l0+0)];
                    val += gout31 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout37 * dm[(j0+0)*nao+(l0+0)];
                    val += gout43 * dm[(j0+1)*nao+(l0+0)];
                    val += gout49 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+1)*nao+(l0+0)];
                    val += gout14 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout20 * dm[(j0+0)*nao+(l0+0)];
                    val += gout26 * dm[(j0+1)*nao+(l0+0)];
                    val += gout32 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout38 * dm[(j0+0)*nao+(l0+0)];
                    val += gout44 * dm[(j0+1)*nao+(l0+0)];
                    val += gout50 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+1)*nao+(l0+0)];
                    val += gout15 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout21 * dm[(j0+0)*nao+(l0+0)];
                    val += gout27 * dm[(j0+1)*nao+(l0+0)];
                    val += gout33 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout39 * dm[(j0+0)*nao+(l0+0)];
                    val += gout45 * dm[(j0+1)*nao+(l0+0)];
                    val += gout51 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+1)*nao+(l0+0)];
                    val += gout16 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout22 * dm[(j0+0)*nao+(l0+0)];
                    val += gout28 * dm[(j0+1)*nao+(l0+0)];
                    val += gout34 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout40 * dm[(j0+0)*nao+(l0+0)];
                    val += gout46 * dm[(j0+1)*nao+(l0+0)];
                    val += gout52 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+1)*nao+(l0+0)];
                    val += gout17 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout23 * dm[(j0+0)*nao+(l0+0)];
                    val += gout29 * dm[(j0+1)*nao+(l0+0)];
                    val += gout35 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout41 * dm[(j0+0)*nao+(l0+0)];
                    val += gout47 * dm[(j0+1)*nao+(l0+0)];
                    val += gout53 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(l0+0)];
                    val += gout19 * dm[(i0+1)*nao+(l0+0)];
                    val += gout20 * dm[(i0+2)*nao+(l0+0)];
                    val += gout21 * dm[(i0+3)*nao+(l0+0)];
                    val += gout22 * dm[(i0+4)*nao+(l0+0)];
                    val += gout23 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout36 * dm[(i0+0)*nao+(l0+0)];
                    val += gout37 * dm[(i0+1)*nao+(l0+0)];
                    val += gout38 * dm[(i0+2)*nao+(l0+0)];
                    val += gout39 * dm[(i0+3)*nao+(l0+0)];
                    val += gout40 * dm[(i0+4)*nao+(l0+0)];
                    val += gout41 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    val += gout9 * dm[(i0+3)*nao+(l0+0)];
                    val += gout10 * dm[(i0+4)*nao+(l0+0)];
                    val += gout11 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout24 * dm[(i0+0)*nao+(l0+0)];
                    val += gout25 * dm[(i0+1)*nao+(l0+0)];
                    val += gout26 * dm[(i0+2)*nao+(l0+0)];
                    val += gout27 * dm[(i0+3)*nao+(l0+0)];
                    val += gout28 * dm[(i0+4)*nao+(l0+0)];
                    val += gout29 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout42 * dm[(i0+0)*nao+(l0+0)];
                    val += gout43 * dm[(i0+1)*nao+(l0+0)];
                    val += gout44 * dm[(i0+2)*nao+(l0+0)];
                    val += gout45 * dm[(i0+3)*nao+(l0+0)];
                    val += gout46 * dm[(i0+4)*nao+(l0+0)];
                    val += gout47 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+1)*nao+(l0+0)];
                    val += gout14 * dm[(i0+2)*nao+(l0+0)];
                    val += gout15 * dm[(i0+3)*nao+(l0+0)];
                    val += gout16 * dm[(i0+4)*nao+(l0+0)];
                    val += gout17 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout30 * dm[(i0+0)*nao+(l0+0)];
                    val += gout31 * dm[(i0+1)*nao+(l0+0)];
                    val += gout32 * dm[(i0+2)*nao+(l0+0)];
                    val += gout33 * dm[(i0+3)*nao+(l0+0)];
                    val += gout34 * dm[(i0+4)*nao+(l0+0)];
                    val += gout35 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout48 * dm[(i0+0)*nao+(l0+0)];
                    val += gout49 * dm[(i0+1)*nao+(l0+0)];
                    val += gout50 * dm[(i0+2)*nao+(l0+0)];
                    val += gout51 * dm[(i0+3)*nao+(l0+0)];
                    val += gout52 * dm[(i0+4)*nao+(l0+0)];
                    val += gout53 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2110(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2110(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2200(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double gout18;
    double gout19;
    double gout20;
    double gout21;
    double gout22;
    double gout23;
    double gout24;
    double gout25;
    double gout26;
    double gout27;
    double gout28;
    double gout29;
    double gout30;
    double gout31;
    double gout32;
    double gout33;
    double gout34;
    double gout35;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        gout18 = 0;
        gout19 = 0;
        gout20 = 0;
        gout21 = 0;
        gout22 = 0;
        gout23 = 0;
        gout24 = 0;
        gout25 = 0;
        gout26 = 0;
        gout27 = 0;
        gout28 = 0;
        gout29 = 0;
        gout30 = 0;
        gout31 = 0;
        gout32 = 0;
        gout33 = 0;
        gout34 = 0;
        gout35 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        double hrr_3100x = trr_40x - xjxi * trr_30x;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        double hrr_2200x = hrr_3100x - xjxi * hrr_2100x;
                        gout0 += hrr_2200x * 1 * wt;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += hrr_1200x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += hrr_1200x * 1 * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        gout3 += hrr_0200x * trr_20y * wt;
                        gout4 += hrr_0200x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout5 += hrr_0200x * 1 * trr_20z;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gout6 += hrr_2100x * hrr_0100y * wt;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout7 += hrr_1100x * hrr_1100y * wt;
                        gout8 += hrr_1100x * hrr_0100y * trr_10z;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        gout9 += hrr_0100x * hrr_2100y * wt;
                        gout10 += hrr_0100x * hrr_1100y * trr_10z;
                        gout11 += hrr_0100x * hrr_0100y * trr_20z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout12 += hrr_2100x * 1 * hrr_0100z;
                        gout13 += hrr_1100x * trr_10y * hrr_0100z;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout14 += hrr_1100x * 1 * hrr_1100z;
                        gout15 += hrr_0100x * trr_20y * hrr_0100z;
                        gout16 += hrr_0100x * trr_10y * hrr_1100z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        gout17 += hrr_0100x * 1 * hrr_2100z;
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        gout18 += trr_20x * hrr_0200y * wt;
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        gout19 += trr_10x * hrr_1200y * wt;
                        gout20 += trr_10x * hrr_0200y * trr_10z;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double hrr_3100y = trr_40y - yjyi * trr_30y;
                        double hrr_2200y = hrr_3100y - yjyi * hrr_2100y;
                        gout21 += 1 * hrr_2200y * wt;
                        gout22 += 1 * hrr_1200y * trr_10z;
                        gout23 += 1 * hrr_0200y * trr_20z;
                        gout24 += trr_20x * hrr_0100y * hrr_0100z;
                        gout25 += trr_10x * hrr_1100y * hrr_0100z;
                        gout26 += trr_10x * hrr_0100y * hrr_1100z;
                        gout27 += 1 * hrr_2100y * hrr_0100z;
                        gout28 += 1 * hrr_1100y * hrr_1100z;
                        gout29 += 1 * hrr_0100y * hrr_2100z;
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        gout30 += trr_20x * 1 * hrr_0200z;
                        gout31 += trr_10x * trr_10y * hrr_0200z;
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        gout32 += trr_10x * 1 * hrr_1200z;
                        gout33 += 1 * trr_20y * hrr_0200z;
                        gout34 += 1 * trr_10y * hrr_1200z;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double hrr_3100z = trr_40z - zjzi * trr_30z;
                        double hrr_2200z = hrr_3100z - zjzi * hrr_2100z;
                        gout35 += 1 * 1 * hrr_2200z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout12 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout18 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+3), val);
                    val = 0;
                    val += gout24 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+4), val);
                    val = 0;
                    val += gout30 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+5), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout13 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout19 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+3), val);
                    val = 0;
                    val += gout25 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+4), val);
                    val = 0;
                    val += gout31 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+5), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout14 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout20 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+3), val);
                    val = 0;
                    val += gout26 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+4), val);
                    val = 0;
                    val += gout32 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+5), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+1), val);
                    val = 0;
                    val += gout15 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+2), val);
                    val = 0;
                    val += gout21 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+3), val);
                    val = 0;
                    val += gout27 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+4), val);
                    val = 0;
                    val += gout33 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+5), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout10 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+1), val);
                    val = 0;
                    val += gout16 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+2), val);
                    val = 0;
                    val += gout22 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+3), val);
                    val = 0;
                    val += gout28 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+4), val);
                    val = 0;
                    val += gout34 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+5), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout11 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+1), val);
                    val = 0;
                    val += gout17 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+2), val);
                    val = 0;
                    val += gout23 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+3), val);
                    val = 0;
                    val += gout29 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+4), val);
                    val = 0;
                    val += gout35 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+5), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    val += gout6 * dm[(j0+1)*nao+(i0+0)];
                    val += gout7 * dm[(j0+1)*nao+(i0+1)];
                    val += gout8 * dm[(j0+1)*nao+(i0+2)];
                    val += gout9 * dm[(j0+1)*nao+(i0+3)];
                    val += gout10 * dm[(j0+1)*nao+(i0+4)];
                    val += gout11 * dm[(j0+1)*nao+(i0+5)];
                    val += gout12 * dm[(j0+2)*nao+(i0+0)];
                    val += gout13 * dm[(j0+2)*nao+(i0+1)];
                    val += gout14 * dm[(j0+2)*nao+(i0+2)];
                    val += gout15 * dm[(j0+2)*nao+(i0+3)];
                    val += gout16 * dm[(j0+2)*nao+(i0+4)];
                    val += gout17 * dm[(j0+2)*nao+(i0+5)];
                    val += gout18 * dm[(j0+3)*nao+(i0+0)];
                    val += gout19 * dm[(j0+3)*nao+(i0+1)];
                    val += gout20 * dm[(j0+3)*nao+(i0+2)];
                    val += gout21 * dm[(j0+3)*nao+(i0+3)];
                    val += gout22 * dm[(j0+3)*nao+(i0+4)];
                    val += gout23 * dm[(j0+3)*nao+(i0+5)];
                    val += gout24 * dm[(j0+4)*nao+(i0+0)];
                    val += gout25 * dm[(j0+4)*nao+(i0+1)];
                    val += gout26 * dm[(j0+4)*nao+(i0+2)];
                    val += gout27 * dm[(j0+4)*nao+(i0+3)];
                    val += gout28 * dm[(j0+4)*nao+(i0+4)];
                    val += gout29 * dm[(j0+4)*nao+(i0+5)];
                    val += gout30 * dm[(j0+5)*nao+(i0+0)];
                    val += gout31 * dm[(j0+5)*nao+(i0+1)];
                    val += gout32 * dm[(j0+5)*nao+(i0+2)];
                    val += gout33 * dm[(j0+5)*nao+(i0+3)];
                    val += gout34 * dm[(j0+5)*nao+(i0+4)];
                    val += gout35 * dm[(j0+5)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout6 * dm[(j0+1)*nao+(k0+0)];
                    val += gout12 * dm[(j0+2)*nao+(k0+0)];
                    val += gout18 * dm[(j0+3)*nao+(k0+0)];
                    val += gout24 * dm[(j0+4)*nao+(k0+0)];
                    val += gout30 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+1)*nao+(k0+0)];
                    val += gout13 * dm[(j0+2)*nao+(k0+0)];
                    val += gout19 * dm[(j0+3)*nao+(k0+0)];
                    val += gout25 * dm[(j0+4)*nao+(k0+0)];
                    val += gout31 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+1)*nao+(k0+0)];
                    val += gout14 * dm[(j0+2)*nao+(k0+0)];
                    val += gout20 * dm[(j0+3)*nao+(k0+0)];
                    val += gout26 * dm[(j0+4)*nao+(k0+0)];
                    val += gout32 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+1)*nao+(k0+0)];
                    val += gout15 * dm[(j0+2)*nao+(k0+0)];
                    val += gout21 * dm[(j0+3)*nao+(k0+0)];
                    val += gout27 * dm[(j0+4)*nao+(k0+0)];
                    val += gout33 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+1)*nao+(k0+0)];
                    val += gout16 * dm[(j0+2)*nao+(k0+0)];
                    val += gout22 * dm[(j0+3)*nao+(k0+0)];
                    val += gout28 * dm[(j0+4)*nao+(k0+0)];
                    val += gout34 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout11 * dm[(j0+1)*nao+(k0+0)];
                    val += gout17 * dm[(j0+2)*nao+(k0+0)];
                    val += gout23 * dm[(j0+3)*nao+(k0+0)];
                    val += gout29 * dm[(j0+4)*nao+(k0+0)];
                    val += gout35 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+0)];
                    val += gout7 * dm[(i0+1)*nao+(k0+0)];
                    val += gout8 * dm[(i0+2)*nao+(k0+0)];
                    val += gout9 * dm[(i0+3)*nao+(k0+0)];
                    val += gout10 * dm[(i0+4)*nao+(k0+0)];
                    val += gout11 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(k0+0)];
                    val += gout13 * dm[(i0+1)*nao+(k0+0)];
                    val += gout14 * dm[(i0+2)*nao+(k0+0)];
                    val += gout15 * dm[(i0+3)*nao+(k0+0)];
                    val += gout16 * dm[(i0+4)*nao+(k0+0)];
                    val += gout17 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(k0+0)];
                    val += gout19 * dm[(i0+1)*nao+(k0+0)];
                    val += gout20 * dm[(i0+2)*nao+(k0+0)];
                    val += gout21 * dm[(i0+3)*nao+(k0+0)];
                    val += gout22 * dm[(i0+4)*nao+(k0+0)];
                    val += gout23 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout24 * dm[(i0+0)*nao+(k0+0)];
                    val += gout25 * dm[(i0+1)*nao+(k0+0)];
                    val += gout26 * dm[(i0+2)*nao+(k0+0)];
                    val += gout27 * dm[(i0+3)*nao+(k0+0)];
                    val += gout28 * dm[(i0+4)*nao+(k0+0)];
                    val += gout29 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout30 * dm[(i0+0)*nao+(k0+0)];
                    val += gout31 * dm[(i0+1)*nao+(k0+0)];
                    val += gout32 * dm[(i0+2)*nao+(k0+0)];
                    val += gout33 * dm[(i0+3)*nao+(k0+0)];
                    val += gout34 * dm[(i0+4)*nao+(k0+0)];
                    val += gout35 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    val += gout18 * dm[(j0+3)*nao+(l0+0)];
                    val += gout24 * dm[(j0+4)*nao+(l0+0)];
                    val += gout30 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout7 * dm[(j0+1)*nao+(l0+0)];
                    val += gout13 * dm[(j0+2)*nao+(l0+0)];
                    val += gout19 * dm[(j0+3)*nao+(l0+0)];
                    val += gout25 * dm[(j0+4)*nao+(l0+0)];
                    val += gout31 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+1)*nao+(l0+0)];
                    val += gout14 * dm[(j0+2)*nao+(l0+0)];
                    val += gout20 * dm[(j0+3)*nao+(l0+0)];
                    val += gout26 * dm[(j0+4)*nao+(l0+0)];
                    val += gout32 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+1)*nao+(l0+0)];
                    val += gout15 * dm[(j0+2)*nao+(l0+0)];
                    val += gout21 * dm[(j0+3)*nao+(l0+0)];
                    val += gout27 * dm[(j0+4)*nao+(l0+0)];
                    val += gout33 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+1)*nao+(l0+0)];
                    val += gout16 * dm[(j0+2)*nao+(l0+0)];
                    val += gout22 * dm[(j0+3)*nao+(l0+0)];
                    val += gout28 * dm[(j0+4)*nao+(l0+0)];
                    val += gout34 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+1)*nao+(l0+0)];
                    val += gout17 * dm[(j0+2)*nao+(l0+0)];
                    val += gout23 * dm[(j0+3)*nao+(l0+0)];
                    val += gout29 * dm[(j0+4)*nao+(l0+0)];
                    val += gout35 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    val += gout9 * dm[(i0+3)*nao+(l0+0)];
                    val += gout10 * dm[(i0+4)*nao+(l0+0)];
                    val += gout11 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+1)*nao+(l0+0)];
                    val += gout14 * dm[(i0+2)*nao+(l0+0)];
                    val += gout15 * dm[(i0+3)*nao+(l0+0)];
                    val += gout16 * dm[(i0+4)*nao+(l0+0)];
                    val += gout17 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(l0+0)];
                    val += gout19 * dm[(i0+1)*nao+(l0+0)];
                    val += gout20 * dm[(i0+2)*nao+(l0+0)];
                    val += gout21 * dm[(i0+3)*nao+(l0+0)];
                    val += gout22 * dm[(i0+4)*nao+(l0+0)];
                    val += gout23 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout24 * dm[(i0+0)*nao+(l0+0)];
                    val += gout25 * dm[(i0+1)*nao+(l0+0)];
                    val += gout26 * dm[(i0+2)*nao+(l0+0)];
                    val += gout27 * dm[(i0+3)*nao+(l0+0)];
                    val += gout28 * dm[(i0+4)*nao+(l0+0)];
                    val += gout29 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout30 * dm[(i0+0)*nao+(l0+0)];
                    val += gout31 * dm[(i0+1)*nao+(l0+0)];
                    val += gout32 * dm[(i0+2)*nao+(l0+0)];
                    val += gout33 * dm[(i0+3)*nao+(l0+0)];
                    val += gout34 * dm[(i0+4)*nao+(l0+0)];
                    val += gout35 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+5)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2200(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2200(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_3000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 2; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        gout0 += trr_30x * 1 * wt;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += trr_20x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += trr_20x * 1 * trr_10z;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        gout3 += trr_10x * trr_20y * wt;
                        gout4 += trr_10x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout5 += trr_10x * 1 * trr_20z;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        gout6 += 1 * trr_30y * wt;
                        gout7 += 1 * trr_20y * trr_10z;
                        gout8 += 1 * trr_10y * trr_20z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        gout9 += 1 * 1 * trr_30z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+6)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+7)*nao+(j0+0), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+8)*nao+(j0+0), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+9)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    val += gout6 * dm[(j0+0)*nao+(i0+6)];
                    val += gout7 * dm[(j0+0)*nao+(i0+7)];
                    val += gout8 * dm[(j0+0)*nao+(i0+8)];
                    val += gout9 * dm[(j0+0)*nao+(i0+9)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    val += gout6 * dm[(i0+6)*nao+(k0+0)];
                    val += gout7 * dm[(i0+7)*nao+(k0+0)];
                    val += gout8 * dm[(i0+8)*nao+(k0+0)];
                    val += gout9 * dm[(i0+9)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    val += gout6 * dm[(i0+6)*nao+(l0+0)];
                    val += gout7 * dm[(i0+7)*nao+(l0+0)];
                    val += gout8 * dm[(i0+8)*nao+(l0+0)];
                    val += gout9 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_jk_3000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_3000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_3010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double gout18;
    double gout19;
    double gout20;
    double gout21;
    double gout22;
    double gout23;
    double gout24;
    double gout25;
    double gout26;
    double gout27;
    double gout28;
    double gout29;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        gout18 = 0;
        gout19 = 0;
        gout20 = 0;
        gout21 = 0;
        gout22 = 0;
        gout23 = 0;
        gout24 = 0;
        gout25 = 0;
        gout26 = 0;
        gout27 = 0;
        gout28 = 0;
        gout29 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        gout0 += trr_31x * 1 * wt;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += trr_21x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += trr_21x * 1 * trr_10z;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        gout3 += trr_11x * trr_20y * wt;
                        gout4 += trr_11x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout5 += trr_11x * 1 * trr_20z;
                        double trr_01x = cpx * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        gout6 += trr_01x * trr_30y * wt;
                        gout7 += trr_01x * trr_20y * trr_10z;
                        gout8 += trr_01x * trr_10y * trr_20z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        gout9 += trr_01x * 1 * trr_30z;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gout10 += trr_30x * trr_01y * wt;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        gout11 += trr_20x * trr_11y * wt;
                        gout12 += trr_20x * trr_01y * trr_10z;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout13 += trr_10x * trr_21y * wt;
                        gout14 += trr_10x * trr_11y * trr_10z;
                        gout15 += trr_10x * trr_01y * trr_20z;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        gout16 += 1 * trr_31y * wt;
                        gout17 += 1 * trr_21y * trr_10z;
                        gout18 += 1 * trr_11y * trr_20z;
                        gout19 += 1 * trr_01y * trr_30z;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout20 += trr_30x * 1 * trr_01z;
                        gout21 += trr_20x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout22 += trr_20x * 1 * trr_11z;
                        gout23 += trr_10x * trr_20y * trr_01z;
                        gout24 += trr_10x * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout25 += trr_10x * 1 * trr_21z;
                        gout26 += 1 * trr_30y * trr_01z;
                        gout27 += 1 * trr_20y * trr_11z;
                        gout28 += 1 * trr_10y * trr_21z;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        gout29 += 1 * 1 * trr_31z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+1)];
                    val += gout20 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+1)];
                    val += gout21 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout12 * dm[(l0+0)*nao+(k0+1)];
                    val += gout22 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout13 * dm[(l0+0)*nao+(k0+1)];
                    val += gout23 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout14 * dm[(l0+0)*nao+(k0+1)];
                    val += gout24 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    val += gout15 * dm[(l0+0)*nao+(k0+1)];
                    val += gout25 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    val += gout16 * dm[(l0+0)*nao+(k0+1)];
                    val += gout26 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+6)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    val += gout17 * dm[(l0+0)*nao+(k0+1)];
                    val += gout27 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+7)*nao+(j0+0), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    val += gout18 * dm[(l0+0)*nao+(k0+1)];
                    val += gout28 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+8)*nao+(j0+0), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    val += gout19 * dm[(l0+0)*nao+(k0+1)];
                    val += gout29 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+9)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    val += gout6 * dm[(j0+0)*nao+(i0+6)];
                    val += gout7 * dm[(j0+0)*nao+(i0+7)];
                    val += gout8 * dm[(j0+0)*nao+(i0+8)];
                    val += gout9 * dm[(j0+0)*nao+(i0+9)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(i0+0)];
                    val += gout11 * dm[(j0+0)*nao+(i0+1)];
                    val += gout12 * dm[(j0+0)*nao+(i0+2)];
                    val += gout13 * dm[(j0+0)*nao+(i0+3)];
                    val += gout14 * dm[(j0+0)*nao+(i0+4)];
                    val += gout15 * dm[(j0+0)*nao+(i0+5)];
                    val += gout16 * dm[(j0+0)*nao+(i0+6)];
                    val += gout17 * dm[(j0+0)*nao+(i0+7)];
                    val += gout18 * dm[(j0+0)*nao+(i0+8)];
                    val += gout19 * dm[(j0+0)*nao+(i0+9)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout20 * dm[(j0+0)*nao+(i0+0)];
                    val += gout21 * dm[(j0+0)*nao+(i0+1)];
                    val += gout22 * dm[(j0+0)*nao+(i0+2)];
                    val += gout23 * dm[(j0+0)*nao+(i0+3)];
                    val += gout24 * dm[(j0+0)*nao+(i0+4)];
                    val += gout25 * dm[(j0+0)*nao+(i0+5)];
                    val += gout26 * dm[(j0+0)*nao+(i0+6)];
                    val += gout27 * dm[(j0+0)*nao+(i0+7)];
                    val += gout28 * dm[(j0+0)*nao+(i0+8)];
                    val += gout29 * dm[(j0+0)*nao+(i0+9)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+0)*nao+(k0+1)];
                    val += gout20 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    val += gout21 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout12 * dm[(j0+0)*nao+(k0+1)];
                    val += gout22 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout13 * dm[(j0+0)*nao+(k0+1)];
                    val += gout23 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout14 * dm[(j0+0)*nao+(k0+1)];
                    val += gout24 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout15 * dm[(j0+0)*nao+(k0+1)];
                    val += gout25 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+0)];
                    val += gout16 * dm[(j0+0)*nao+(k0+1)];
                    val += gout26 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(k0+0)];
                    val += gout17 * dm[(j0+0)*nao+(k0+1)];
                    val += gout27 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(k0+0)];
                    val += gout18 * dm[(j0+0)*nao+(k0+1)];
                    val += gout28 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    val += gout19 * dm[(j0+0)*nao+(k0+1)];
                    val += gout29 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout10 * dm[(i0+0)*nao+(k0+1)];
                    val += gout20 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout11 * dm[(i0+1)*nao+(k0+1)];
                    val += gout21 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout12 * dm[(i0+2)*nao+(k0+1)];
                    val += gout22 * dm[(i0+2)*nao+(k0+2)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout13 * dm[(i0+3)*nao+(k0+1)];
                    val += gout23 * dm[(i0+3)*nao+(k0+2)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout14 * dm[(i0+4)*nao+(k0+1)];
                    val += gout24 * dm[(i0+4)*nao+(k0+2)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    val += gout15 * dm[(i0+5)*nao+(k0+1)];
                    val += gout25 * dm[(i0+5)*nao+(k0+2)];
                    val += gout6 * dm[(i0+6)*nao+(k0+0)];
                    val += gout16 * dm[(i0+6)*nao+(k0+1)];
                    val += gout26 * dm[(i0+6)*nao+(k0+2)];
                    val += gout7 * dm[(i0+7)*nao+(k0+0)];
                    val += gout17 * dm[(i0+7)*nao+(k0+1)];
                    val += gout27 * dm[(i0+7)*nao+(k0+2)];
                    val += gout8 * dm[(i0+8)*nao+(k0+0)];
                    val += gout18 * dm[(i0+8)*nao+(k0+1)];
                    val += gout28 * dm[(i0+8)*nao+(k0+2)];
                    val += gout9 * dm[(i0+9)*nao+(k0+0)];
                    val += gout19 * dm[(i0+9)*nao+(k0+1)];
                    val += gout29 * dm[(i0+9)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout20 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout21 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout22 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout23 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout24 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout25 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+6)*nao+(k0+1), val);
                    val = 0;
                    val += gout26 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+6)*nao+(k0+2), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                    val = 0;
                    val += gout17 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+7)*nao+(k0+1), val);
                    val = 0;
                    val += gout27 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+7)*nao+(k0+2), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+8)*nao+(k0+1), val);
                    val = 0;
                    val += gout28 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+8)*nao+(k0+2), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                    val = 0;
                    val += gout19 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+9)*nao+(k0+1), val);
                    val = 0;
                    val += gout29 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+9)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    val += gout6 * dm[(i0+6)*nao+(l0+0)];
                    val += gout7 * dm[(i0+7)*nao+(l0+0)];
                    val += gout8 * dm[(i0+8)*nao+(l0+0)];
                    val += gout9 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(l0+0)];
                    val += gout11 * dm[(i0+1)*nao+(l0+0)];
                    val += gout12 * dm[(i0+2)*nao+(l0+0)];
                    val += gout13 * dm[(i0+3)*nao+(l0+0)];
                    val += gout14 * dm[(i0+4)*nao+(l0+0)];
                    val += gout15 * dm[(i0+5)*nao+(l0+0)];
                    val += gout16 * dm[(i0+6)*nao+(l0+0)];
                    val += gout17 * dm[(i0+7)*nao+(l0+0)];
                    val += gout18 * dm[(i0+8)*nao+(l0+0)];
                    val += gout19 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout20 * dm[(i0+0)*nao+(l0+0)];
                    val += gout21 * dm[(i0+1)*nao+(l0+0)];
                    val += gout22 * dm[(i0+2)*nao+(l0+0)];
                    val += gout23 * dm[(i0+3)*nao+(l0+0)];
                    val += gout24 * dm[(i0+4)*nao+(l0+0)];
                    val += gout25 * dm[(i0+5)*nao+(l0+0)];
                    val += gout26 * dm[(i0+6)*nao+(l0+0)];
                    val += gout27 * dm[(i0+7)*nao+(l0+0)];
                    val += gout28 * dm[(i0+8)*nao+(l0+0)];
                    val += gout29 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_3010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_3010(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_3100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double gout18;
    double gout19;
    double gout20;
    double gout21;
    double gout22;
    double gout23;
    double gout24;
    double gout25;
    double gout26;
    double gout27;
    double gout28;
    double gout29;
    double val;
    double *dm, *vj, *vk;

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        gout18 = 0;
        gout19 = 0;
        gout20 = 0;
        gout21 = 0;
        gout22 = 0;
        gout23 = 0;
        gout24 = 0;
        gout25 = 0;
        gout26 = 0;
        gout27 = 0;
        gout28 = 0;
        gout29 = 0;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw+6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, rw1+nsq_per_block, 2*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, rw+nsq_per_block, 2*nsq_per_block);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 3; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        double hrr_3100x = trr_40x - xjxi * trr_30x;
                        gout0 += hrr_3100x * 1 * wt;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        gout1 += hrr_2100x * trr_10y * wt;
                        double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout2 += hrr_2100x * 1 * trr_10z;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        gout3 += hrr_1100x * trr_20y * wt;
                        gout4 += hrr_1100x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout5 += hrr_1100x * 1 * trr_20z;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        gout6 += hrr_0100x * trr_30y * wt;
                        gout7 += hrr_0100x * trr_20y * trr_10z;
                        gout8 += hrr_0100x * trr_10y * trr_20z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        gout9 += hrr_0100x * 1 * trr_30z;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gout10 += trr_30x * hrr_0100y * wt;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout11 += trr_20x * hrr_1100y * wt;
                        gout12 += trr_20x * hrr_0100y * trr_10z;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        gout13 += trr_10x * hrr_2100y * wt;
                        gout14 += trr_10x * hrr_1100y * trr_10z;
                        gout15 += trr_10x * hrr_0100y * trr_20z;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double hrr_3100y = trr_40y - yjyi * trr_30y;
                        gout16 += 1 * hrr_3100y * wt;
                        gout17 += 1 * hrr_2100y * trr_10z;
                        gout18 += 1 * hrr_1100y * trr_20z;
                        gout19 += 1 * hrr_0100y * trr_30z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout20 += trr_30x * 1 * hrr_0100z;
                        gout21 += trr_20x * trr_10y * hrr_0100z;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout22 += trr_20x * 1 * hrr_1100z;
                        gout23 += trr_10x * trr_20y * hrr_0100z;
                        gout24 += trr_10x * trr_10y * hrr_1100z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        gout25 += trr_10x * 1 * hrr_2100z;
                        gout26 += 1 * trr_30y * hrr_0100z;
                        gout27 += 1 * trr_20y * hrr_1100z;
                        gout28 += 1 * trr_10y * hrr_2100z;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double hrr_3100z = trr_40z - zjzi * trr_30z;
                        gout29 += 1 * 1 * hrr_3100z;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout10 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout20 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout11 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout21 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout12 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout22 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout13 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+1), val);
                    val = 0;
                    val += gout23 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+2), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout14 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+1), val);
                    val = 0;
                    val += gout24 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+2), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout15 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+1), val);
                    val = 0;
                    val += gout25 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+2), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+6)*nao+(j0+0), val);
                    val = 0;
                    val += gout16 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+6)*nao+(j0+1), val);
                    val = 0;
                    val += gout26 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+6)*nao+(j0+2), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+7)*nao+(j0+0), val);
                    val = 0;
                    val += gout17 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+7)*nao+(j0+1), val);
                    val = 0;
                    val += gout27 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+7)*nao+(j0+2), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+8)*nao+(j0+0), val);
                    val = 0;
                    val += gout18 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+8)*nao+(j0+1), val);
                    val = 0;
                    val += gout28 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+8)*nao+(j0+2), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+9)*nao+(j0+0), val);
                    val = 0;
                    val += gout19 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+9)*nao+(j0+1), val);
                    val = 0;
                    val += gout29 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+9)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    val += gout6 * dm[(j0+0)*nao+(i0+6)];
                    val += gout7 * dm[(j0+0)*nao+(i0+7)];
                    val += gout8 * dm[(j0+0)*nao+(i0+8)];
                    val += gout9 * dm[(j0+0)*nao+(i0+9)];
                    val += gout10 * dm[(j0+1)*nao+(i0+0)];
                    val += gout11 * dm[(j0+1)*nao+(i0+1)];
                    val += gout12 * dm[(j0+1)*nao+(i0+2)];
                    val += gout13 * dm[(j0+1)*nao+(i0+3)];
                    val += gout14 * dm[(j0+1)*nao+(i0+4)];
                    val += gout15 * dm[(j0+1)*nao+(i0+5)];
                    val += gout16 * dm[(j0+1)*nao+(i0+6)];
                    val += gout17 * dm[(j0+1)*nao+(i0+7)];
                    val += gout18 * dm[(j0+1)*nao+(i0+8)];
                    val += gout19 * dm[(j0+1)*nao+(i0+9)];
                    val += gout20 * dm[(j0+2)*nao+(i0+0)];
                    val += gout21 * dm[(j0+2)*nao+(i0+1)];
                    val += gout22 * dm[(j0+2)*nao+(i0+2)];
                    val += gout23 * dm[(j0+2)*nao+(i0+3)];
                    val += gout24 * dm[(j0+2)*nao+(i0+4)];
                    val += gout25 * dm[(j0+2)*nao+(i0+5)];
                    val += gout26 * dm[(j0+2)*nao+(i0+6)];
                    val += gout27 * dm[(j0+2)*nao+(i0+7)];
                    val += gout28 * dm[(j0+2)*nao+(i0+8)];
                    val += gout29 * dm[(j0+2)*nao+(i0+9)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+1)*nao+(k0+0)];
                    val += gout20 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout11 * dm[(j0+1)*nao+(k0+0)];
                    val += gout21 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout12 * dm[(j0+1)*nao+(k0+0)];
                    val += gout22 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout13 * dm[(j0+1)*nao+(k0+0)];
                    val += gout23 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout14 * dm[(j0+1)*nao+(k0+0)];
                    val += gout24 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout15 * dm[(j0+1)*nao+(k0+0)];
                    val += gout25 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+0)];
                    val += gout16 * dm[(j0+1)*nao+(k0+0)];
                    val += gout26 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(k0+0)];
                    val += gout17 * dm[(j0+1)*nao+(k0+0)];
                    val += gout27 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(k0+0)];
                    val += gout18 * dm[(j0+1)*nao+(k0+0)];
                    val += gout28 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    val += gout19 * dm[(j0+1)*nao+(k0+0)];
                    val += gout29 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    val += gout6 * dm[(i0+6)*nao+(k0+0)];
                    val += gout7 * dm[(i0+7)*nao+(k0+0)];
                    val += gout8 * dm[(i0+8)*nao+(k0+0)];
                    val += gout9 * dm[(i0+9)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(k0+0)];
                    val += gout11 * dm[(i0+1)*nao+(k0+0)];
                    val += gout12 * dm[(i0+2)*nao+(k0+0)];
                    val += gout13 * dm[(i0+3)*nao+(k0+0)];
                    val += gout14 * dm[(i0+4)*nao+(k0+0)];
                    val += gout15 * dm[(i0+5)*nao+(k0+0)];
                    val += gout16 * dm[(i0+6)*nao+(k0+0)];
                    val += gout17 * dm[(i0+7)*nao+(k0+0)];
                    val += gout18 * dm[(i0+8)*nao+(k0+0)];
                    val += gout19 * dm[(i0+9)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout20 * dm[(i0+0)*nao+(k0+0)];
                    val += gout21 * dm[(i0+1)*nao+(k0+0)];
                    val += gout22 * dm[(i0+2)*nao+(k0+0)];
                    val += gout23 * dm[(i0+3)*nao+(k0+0)];
                    val += gout24 * dm[(i0+4)*nao+(k0+0)];
                    val += gout25 * dm[(i0+5)*nao+(k0+0)];
                    val += gout26 * dm[(i0+6)*nao+(k0+0)];
                    val += gout27 * dm[(i0+7)*nao+(k0+0)];
                    val += gout28 * dm[(i0+8)*nao+(k0+0)];
                    val += gout29 * dm[(i0+9)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+1)*nao+(l0+0)];
                    val += gout20 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+1)*nao+(l0+0)];
                    val += gout21 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+1)*nao+(l0+0)];
                    val += gout22 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout13 * dm[(j0+1)*nao+(l0+0)];
                    val += gout23 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout14 * dm[(j0+1)*nao+(l0+0)];
                    val += gout24 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout15 * dm[(j0+1)*nao+(l0+0)];
                    val += gout25 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    val += gout16 * dm[(j0+1)*nao+(l0+0)];
                    val += gout26 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    val += gout17 * dm[(j0+1)*nao+(l0+0)];
                    val += gout27 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    val += gout18 * dm[(j0+1)*nao+(l0+0)];
                    val += gout28 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    val += gout19 * dm[(j0+1)*nao+(l0+0)];
                    val += gout29 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    val += gout6 * dm[(i0+6)*nao+(l0+0)];
                    val += gout7 * dm[(i0+7)*nao+(l0+0)];
                    val += gout8 * dm[(i0+8)*nao+(l0+0)];
                    val += gout9 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(l0+0)];
                    val += gout11 * dm[(i0+1)*nao+(l0+0)];
                    val += gout12 * dm[(i0+2)*nao+(l0+0)];
                    val += gout13 * dm[(i0+3)*nao+(l0+0)];
                    val += gout14 * dm[(i0+4)*nao+(l0+0)];
                    val += gout15 * dm[(i0+5)*nao+(l0+0)];
                    val += gout16 * dm[(i0+6)*nao+(l0+0)];
                    val += gout17 * dm[(i0+7)*nao+(l0+0)];
                    val += gout18 * dm[(i0+8)*nao+(l0+0)];
                    val += gout19 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout20 * dm[(i0+0)*nao+(l0+0)];
                    val += gout21 * dm[(i0+1)*nao+(l0+0)];
                    val += gout22 * dm[(i0+2)*nao+(l0+0)];
                    val += gout23 * dm[(i0+3)*nao+(l0+0)];
                    val += gout24 * dm[(i0+4)*nao+(l0+0)];
                    val += gout25 * dm[(i0+5)*nao+(l0+0)];
                    val += gout26 * dm[(i0+6)*nao+(l0+0)];
                    val += gout27 * dm[(i0+7)*nao+(l0+0)];
                    val += gout28 * dm[(i0+8)*nao+(l0+0)];
                    val += gout29 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_3100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_3100(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

int rys_jk_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head,
                    int *scheme, int workers, double omega)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int ijkl = li*125 + lj*25 + lk*5 + ll;
    int nroots = (li + lj + lk + ll) / 2 + 1;
    int g_size = bounds->stride_l * (bounds->ll + 1);
    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int ij_prims = iprim * jprim;
    int buflen = ij_prims*TILE2;
    int nsq_per_block = scheme[0];
    int gout_stride = scheme[1];

    switch (ijkl) {
    case 0:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 125:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 130:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 131:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 150:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 155:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 156:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 250:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 255:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 256:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 260:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 275:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 280:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 300:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 375:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 380:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    case 400:
        nsq_per_block = 256;
        gout_stride = 1;
        break;
    }

#if CUDA_VERSION >= 12040
    switch (ijkl) {
    case 0: nsq_per_block *= 2; break;
    case 125: nsq_per_block *= 2; break;
    case 130: nsq_per_block *= 2; break;
    case 150: nsq_per_block *= 2; break;
    case 250: nsq_per_block *= 2; break;
    case 255: nsq_per_block *= 2; break;
    case 275: nsq_per_block *= 2; break;
    case 375: nsq_per_block *= 2; break;
    }
#endif

    dim3 threads(nsq_per_block, gout_stride);
    buflen += nroots*2 * nsq_per_block;
    if (omega < 0) {
        buflen += nroots*2 * nsq_per_block;
    }
    switch (ijkl) {
    case 0:
        buflen += ij_prims*TILE2*3;
        rys_jk_0000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 125:
        buflen += ij_prims*TILE2*3;
        rys_jk_1000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 130:
        buflen += ij_prims*TILE2*3;
        rys_jk_1010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 131:
        buflen += ij_prims*TILE2*3;
        rys_jk_1011<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 150:
        buflen += ij_prims*TILE2*3;
        rys_jk_1100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 155:
        buflen += ij_prims*TILE2*3;
        rys_jk_1110<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 156:
        buflen += ij_prims*TILE2*3;
        rys_jk_1111<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 250:
        buflen += ij_prims*TILE2*3;
        rys_jk_2000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 255:
        buflen += ij_prims*TILE2*3;
        rys_jk_2010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 256:
        buflen += ij_prims*TILE2*3;
        rys_jk_2011<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 260:
        buflen += ij_prims*TILE2*3;
        rys_jk_2020<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 275:
        buflen += ij_prims*TILE2*3;
        rys_jk_2100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 280:
        buflen += ij_prims*TILE2*3;
        rys_jk_2110<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 300:
        buflen += ij_prims*TILE2*3;
        rys_jk_2200<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 375:
        buflen += ij_prims*TILE2*3;
        rys_jk_3000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 380:
        buflen += ij_prims*TILE2*3;
        rys_jk_3010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 400:
        buflen += ij_prims*TILE2*3;
        rys_jk_3100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    default: return 0;
    }
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess;
}
