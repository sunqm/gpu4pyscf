# gpu4pyscf is a plugin to use Nvidia GPU in PySCF pacakge
#
# Copyright (C) 2022 Qiming Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import time
import ctypes
import cupy
import numpy as np
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo.outcore import balance_partition
from pyscf.ao2mo import _ao2mo
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf import __config__
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.lib.utils import patch_cpu_kernel

FREE_CUPY_CACHE = True

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

libgint = lib.load_library('libgint')
libgint.GINTfill_int2e.restype = ctypes.c_int

@patch_cpu_kernel(ccsd.update_amps)
def update_amps(mycc, t1, t2, eris):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    orbo = eris.mo_coeff[:,:nocc]
    orbv = eris.mo_coeff[:,nocc:]

    wpq, t1new, t2new, wVOov, wVooV = _direct_ovvv_vvvv(mycc, t1, t2)
    t2new *= .5  # *.5 because t2+t2.transpose(1,0,3,2) at the end
    time1 = log.timer_debug1('vvvv', *time0)

    fov = fock[:nocc,nocc:].copy()
    t1new += fock[:nocc,nocc:]

    foo = fock[:nocc,:nocc] - np.diag(mo_e_o)
    foo += .5 * np.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)

    fvv = lib.einsum('pa,qp,qb->ab', orbv, wpq, orbv)
    t1new -= lib.einsum('ab,ib->ia', fvv, t1)

    fvv += fock[nocc:,nocc:] - np.diag(mo_e_v)
    fvv -= .5 * np.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])

    foo += lib.einsum('pi,qp,qj->ij', orbo, wpq, orbo)
    fov += lib.einsum('pi,qp,qa->ia', orbo, wpq, orbv)

    tau = np.einsum('ia,jb->ijab', t1, t1)
    tau += t2
    woooo = lib.einsum('ijab,kabl->ijkl', tau, eris.ovvo)
    woooo += eris.oooo.transpose(0,2,1,3)
    tmp = lib.einsum('la,jaik->lkji', t1, eris.ovoo)
    woooo += tmp
    woooo += tmp.transpose(1,0,3,2)
    t2new += .5 * lib.einsum('ijkl,klab->ijab', woooo, tau)
    woooo = tau = None

    wVOov -= lib.einsum('jbik,ka->bjia', eris.ovoo, t1)
    t2new += wVOov.transpose(1,2,0,3)

    wVooV += lib.einsum('kbij,ka->bija', eris.ovoo, t1)
    wVooV -= eris.oovv.transpose(2,0,1,3)
    wVOov += wVooV*.5  #: bjia + bija*.5
    wVOov += eris.ovvo.transpose(2,3,0,1)

    t2new += (eris.ovvo*0.5).transpose(0,3,1,2)
    t1new += lib.einsum('pi,pq,qa->ia', orbo, wpq, orbv)

    tmp  = lib.einsum('ic,kjbc->ikjb', t1, eris.oovv)
    tmp += lib.einsum('jbck,ic->jkib', eris.ovvo, t1)
    t2new -= lib.einsum('ka,jkib->jiba', t1, tmp)
    tmp = None

    tau  = t2 * .5
    tau += np.einsum('ia,jb->ijab', t1, t1)
    wVooV += lib.einsum('kbci,jkca->bija', eris.ovvo, tau)
    tau = None

    tmp = lib.einsum('jkca,ckib->jaib', t2, wVooV)
    t2new += tmp.transpose(2,0,1,3)
    tmp *= .5
    t2new += tmp.transpose(0,2,1,3)
    tmp = None

    tau  = np.einsum('ia,jb->iajb', t1*.5, t1)
    tau += t2.transpose(0,2,1,3)
    eris_ovOV = eris.ovvo.transpose(0,1,3,2) * 2
    eris_ovOV -= eris.ovvo.transpose(3,1,0,2)
    fvv -= lib.einsum('jcia,jcib->ab', tau, eris_ovOV)
    foo += lib.einsum('iakb,jakb->ij', eris_ovOV, tau)

    theta  = t2.transpose(0,2,1,3) * 2
    theta -= t2.transpose(1,2,0,3)
    tau = theta * .25
    tau -= np.einsum('ia,jb->jaib', t1*.5, t1)
    wVOov += lib.einsum('kcia,kcjb->aijb', eris_ovOV, tau)
    eris_ovOV = tau = None

    t2new += lib.einsum('kcia,ckjb->ijab', theta, wVOov)
    theta = wVOov = wVooV = None

    t1new += np.einsum('jb,ijab->ia', fov, t2) * 2
    t1new -= np.einsum('jb,ijba->ia', fov, t2)
    ovoo = eris.ovoo * 2
    ovoo -= eris.ovoo.transpose(2,1,0,3)
    t1new -= lib.einsum('jbki,jkba->ia', ovoo, t2)
    ovoo = None

    ft_ij = foo + np.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - np.einsum('ia,ib->ab', .5*t1, fov)
    t2new += lib.einsum('ijac,bc->ijab', t2, ft_ab)
    t2new -= lib.einsum('ki,kjab->ijab', ft_ij, t2)

    eia = mo_e_o[:,None] - mo_e_v
    t1new += np.einsum('ib,ab->ia', t1, fvv)
    t1new -= np.einsum('ja,ji->ia', t1, foo)
    t1new /= eia

    #: t2new = t2new + t2new.transpose(1,0,3,2)
    for i in range(nocc):
        if i > 0:
            t2new[i,:i] += t2new[:i,i].transpose(0,2,1)
            t2new[i,:i] /= lib.direct_sum('a,jb->jab', eia[i], eia[:i])
            t2new[:i,i] = t2new[i,:i].transpose(0,2,1)
        t2new[i,i] = t2new[i,i] + t2new[i,i].T
        t2new[i,i] /= lib.direct_sum('a,b->ab', eia[i], eia[i])

    time0 = log.timer_debug1('update t1 t2', *time0)
    return t1new, t2new

def _direct_ovvv_vvvv(mycc, t1, t2):
    if getattr(mycc, 'device', 'cpu') == 'gpu':
        return _direct_ovvv_vvvv_gpu(mycc, t1, t2)
    else:
        return _direct_ovvv_vvvv_cpu(mycc, t1, t2)

# Memory footprint: roughly t2.size * 4 + eri_buf.size * 2
def _direct_ovvv_vvvv_cpu(mycc, t1, t2):
    mol = mycc.mol
    nocc, nvir = t1.shape
    nocc2 = nocc*(nocc+1)//2
    mo = mycc.mo_coeff
    nao, nmo = mo.shape
    aos = np.asarray(mo[:,nocc:].T, order='F')
    tau = make_tau_tril(t1, t2)
    tau = _ao2mo.nr_e2(tau.reshape(nocc2,nvir**2), aos, (0,nao,0,nao), 's1', 's1')
    x2 = tau.reshape(nocc2,nao,nao)
    tau = None

    ao_loc = mol.ao_loc
    nao2 = nao * nao
    _einsum = np.einsum
    _dot = lib.ddot
    orbo = np.asarray(mycc.mo_coeff[:,:nocc])
    orbv = np.asarray(mycc.mo_coeff[:,nocc:])
    t1po = orbv.dot(t1.T)
    Ht2ao = np.zeros(x2.shape)
    _dgemm = lib.numpy_helper._dgemm
    def contract_vvvv_(eri, i0, i1, j0, j1):
        ic = i1 - i0
        jc = j1 - j0
        #:Ht2[:,j0:j1] += np.einsum('xef,efab->xab', x2[:,i0:i1], eri)
        _dgemm('N', 'N', nocc2, jc*nao, ic*nao,
               x2.reshape(-1,nao2), eri.reshape(-1,jc*nao),
               Ht2ao.reshape(-1,nao2), 1, 1, i0*nao, 0, j0*nao)

        if i0 > j0:
            #:Ht2[:,i0:i1] += np.einsum('xef,abef->xab', x2[:,j0:j1], eri)
            _dgemm('N', 'T', nocc2, ic*nao, jc*nao,
                   x2.reshape(-1,nao2), eri.reshape(-1,jc*nao),
                   Ht2ao.reshape(-1,nao2), 1, 1, j0*nao, 0, i0*nao)

    intor = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    max_memory = max(MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    blksize = max(BLKMIN, ((max_memory*.9e6-t2.size*4*8)/8/nao**2/3.5)**.5)
    blksize = int(min((nao+3)/4, blksize))
    sh_ranges = balance_partition(ao_loc, blksize)
    blksize = max(x[2] for x in sh_ranges)

    eribuf = np.empty((blksize,blksize,nao,nao))
    loadbuf = np.empty((blksize,blksize,nao,nao))
    def fint(ish0, ish1, jsh0, jsh1, group_id):
        if ish0 != jsh0:
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
            eri = gto.moleintor.getints4c(
                intor, mol._atm, mol._bas, mol._env,
                shls_slice=(ish0,ish1,jsh0,jsh1), aosym='s2kl',
                ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
            aoblk = np.ndarray((i1-i0,nao,j1-j0,nao), buffer=loadbuf)
            _ccsd.libcc.CCload_eri(aoblk.ctypes.data_as(ctypes.c_void_p),
                                   eri.ctypes.data_as(ctypes.c_void_p),
                                   (ctypes.c_int*4)(i0, i1, j0, j1),
                                   ctypes.c_int(nao))
        else:
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            eri = gto.moleintor.getints4c(
                intor, mol._atm, mol._bas, mol._env,
                shls_slice=(ish0,ish1,ish0,ish1), aosym='s4',
                ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
            eri = lib.unpack_tril(eri, axis=0)
            aoblk = np.ndarray((i1-i0,nao,i1-i0,nao), buffer=loadbuf)
            _ccsd.libcc.CCload_eri(aoblk.ctypes.data_as(ctypes.c_void_p),
                                   eri.ctypes.data_as(ctypes.c_void_p),
                                   (ctypes.c_int*4)(i0, i1, i0, i1),
                                   ctypes.c_int(nao))
        return aoblk

    wVVoo = np.zeros((nao,nao,nocc,nocc))
    wVvoO = np.zeros((nao,nao,nocc,nocc))

    for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
        for jsh0, jsh1, nj in sh_ranges[:ip]:
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
            aoblk = fint(ish0, ish1, jsh0, jsh1, None)
            contract_vvvv_(aoblk, i0, i1, j0, j1)

            #:fvv += 2*np.einsum('kc,kcab->ab', t1, eris_ovvv)
            #:fvv -= np.einsum('kc,kbca->ab', t1, eris_ovvv)
            # pppo = _einsum('PrQs,si->PrQi', aoblk, orbo)
            pppo = _dot(aoblk.reshape(-1,nao), orbo).reshape(i1-i0,nao,j1-j0,nocc)
            # tmp = _einsum('PrQi,Pj->rQij', pppo, t1po[i0:i1])
            tmp = _dot(pppo.reshape(i1-i0,-1).T, t1po[i0:i1]).reshape(nao,j1-j0,nocc,nocc)
            wVvoO[j0:j1] += tmp.transpose(1,0,2,3)
            tmp = None
            # pqji = _einsum('PrQi,rj->PQij', pppo, t1po)
            pqji = _dot(pppo.transpose(1,0,2,3).reshape(nao,-1).T,
                        t1po).reshape(i1-i0,j1-j0,nocc,nocc)
            wVVoo[i0:i1,j0:j1] = pqji
            wVVoo[j0:j1,i0:i1] = pqji.transpose(1,0,2,3)
            pppo = pqji = None

            for i in range(i1 - i0):
                tmp = _dot(aoblk[i].reshape(nao,-1).T, orbo)
                wVvoO[i0+i] += _dot(tmp.reshape(j1-j0,-1).T,
                                    t1po[j0:j1]).reshape(nao,nocc,nocc)
            aoblk = tmp = None

        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        aoblk = fint(ish0, ish1, ish0, ish1, None)
        contract_vvvv_(aoblk, i0, i1, i0, i1)

        # pppo = _einsum('PrQs,si->PrQi', aoblk, orbo)
        pppo = _dot(aoblk.reshape(-1,nao), orbo).reshape(i1-i0,nao,i1-i0,nocc)
        # tmp = _einsum('PrQi,Pj->rQij', pppo, t1po[i0:i1])
        tmp = _dot(pppo.reshape(i1-i0,-1).T, t1po[i0:i1]).reshape(nao,i1-i0,nocc,nocc)
        aoblk = None
        wVvoO[i0:i1] += tmp.transpose(1,0,2,3)
        tmp = None
        # pqji = _einsum('PrQi,rj->PQij', pppo, t1po)
        pqji = _dot(pppo.transpose(1,0,2,3).reshape(nao,-1).T,
                    t1po).reshape(i1-i0,i1-i0,nocc,nocc)
        wVVoo[i0:i1,i0:i1] = pqji
        pppo = pqji = None
    eribuf = loadbuf = tau = x2 = None

    #:wVOov = lib.einsum('pqji,qb,pa->bjia', wVvoO, orbv, orbv)
    #:wVooV =-lib.einsum('pqji,pa,qb->bjia', wVVoo, orbv, orbv)
    wpq = np.einsum('pqii->pq', wVVoo) * 2
    qjia = _dot(wVVoo.reshape(nao,-1).T, -orbv)
    wVVoo = None
    wVooV = _dot(orbv.T, qjia.reshape(nao,-1)).reshape(nvir,nocc,nocc,nvir)
    qjia = None

    wpq -= np.einsum('pqii->qp', wVvoO)
    qjia = _dot(wVvoO.reshape(nao,-1).T, orbv)
    wVvoO = None
    wVOov = _dot(orbv.T, qjia.reshape(nao,-1)).reshape(nvir,nocc,nocc,nvir)
    qjia = None

    #:t1new += 2*lib.einsum('edac,ikcd->ikea', eris_ovvv, t2)
    #:t1new +=  -lib.einsum('edac,ikdc->ikea', eris_ovvv, t2)
    Ht2full = ccsd._unpack_t2_tril(Ht2ao, nocc, nao, t2sym='jiba')
    Ht2full = Ht2full.reshape(nocc,nocc,nao,nao)
    t1tmp  = np.einsum('ijpq,qj->ip', Ht2full, orbo) * 2
    t1tmp -= np.einsum('ijqp,qj->ip', Ht2full, orbo)
    t1new = t1tmp.dot(orbv)
    Ht2full = t1tmp = None

    # vvvv-t2 contractions back to MO repr.
    Ht2tril = _ao2mo.nr_e2(Ht2ao, mo.conj(), (nocc,nmo,nocc,nmo), 's1', 's1')
    Ht2tril = Ht2tril.reshape(nocc2,nvir,nvir)

    # part of ovvv-t2 contractions back to MO repr.
    #: tmp = np.einsum('ijcd,ka,kdcb->ijba', tau, t1, eris.ovvv)
    #: t2new -= tmp + tmp.transpose(1,0,3,2)
    tmp = _ao2mo.nr_e2(Ht2ao, mo.conj(), (nocc,nmo,0,nocc), 's1', 's1')
    Ht2tril -= lib.ddot(tmp.reshape(nocc2*nvir,nocc), t1).reshape(nocc2,nvir,nvir)
    tmp = _ao2mo.nr_e2(Ht2ao, mo.conj(), (0,nocc,nocc,nmo), 's1', 's1')
    Ht2ao = None
    #: Ht2tril -= np.einsum('xkb,ka->xab', tmp.reshape(-1,nocc,nvir), t1)
    tmp = lib.transpose(tmp.reshape(nocc2,nocc,nvir), axes=(0,2,1))
    tmp = lib.ddot(tmp.reshape(nocc2*nvir,nocc), t1, 1,
                   np.ndarray((nocc2*nvir,nvir)), 0)
    tmp = lib.transpose(tmp.reshape(nocc2,nvir,nvir), axes=(0,2,1))
    Ht2tril -= tmp.reshape(nocc2,nvir,nvir)

    t2new = ccsd._unpack_t2_tril(Ht2tril, nocc, nvir, t2sym='jiba')

    return wpq, t1new, t2new, wVOov, wVooV

less_gpu_mem = True
def _direct_ovvv_vvvv_gpu(mycc, t1, t2):
    nocc, nvir = t1.shape
    nocc2 = nocc*(nocc+1)//2
    nao_cart = mycc.mol.nao_nr(cart=True)
    max_memory = max(MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    blksize = ((max_memory*.9e6-t2.size*4*8)/8/nao_cart**2/3.5)**.5
    mem_avail = int(cupy.cuda.runtime.memGetInfo()[0] * .5)
    cupy.get_default_memory_pool().set_limit(mem_avail)
    if not less_gpu_mem:
        mem_avail -= nocc2*nao_cart**2 * 8 * 2  # x2 and Ht2
    blksize = max(BLKMIN, int(min((nao_cart+3)/4, blksize,
                                  (mem_avail*.5/8/nao_cart**2)**.5)))
    logger.debug1(mycc, 'blksize %d nao %d', blksize, nao_cart)

    vhfopt = gpu_hf._VHFOpt(mycc.mol, 'int2e')
    vhfopt.build(group_size=blksize, diag_block_with_triu=True)
    mol = vhfopt.mol

    mo = vhfopt.coeff.dot(mycc.mo_coeff)
    nao, nmo = mo.shape
    aos = np.asarray(mo[:,nocc:].T, order='F')
    tau = make_tau_tril(t1, t2)
    tau = _ao2mo.nr_e2(tau.reshape(nocc2,nvir**2), aos, (0,nao,0,nao), 's1', 's1')
    x2 = tau.reshape(nocc2,nao,nao)
    tau = None

    orbo = cupy.asarray(mo[:,:nocc])
    orbv = cupy.asarray(mo[:,nocc:])
    t1po = orbv.dot(cupy.asarray(t1).T)

    ao_loc = mol.ao_loc
    nao2 = nao * nao

    _einsum = cupy.einsum
    _dot = cupy.dot
    if less_gpu_mem:
        x2 = lib.transpose(x2.reshape(nocc2,-1)).reshape(nao,nao,nocc2)
        Ht2ao = np.zeros(x2.shape)
        def contract_vvvv_(eri, i0, i1, j0, j1):
            jc = j1 - j0
            eri = eri.reshape(-1,jc*nao)
            #:Ht2[:,j0:j1] += np.einsum('xef,efab->xab', x2[:,i0:i1], eri)
            x2tmp = cupy.asarray(x2[i0:i1].reshape(-1,nocc2))
            Ht2ao[j0:j1] += _dot(eri.T, x2tmp).reshape(j1-j0,nao,nocc2).get()
            if i0 > j0:
                #:Ht2[:,i0:i1] += np.einsum('xef,abef->xab', x2[:,j0:j1], eri)
                x2tmp = cupy.asarray(x2[j0:j1].reshape(-1,nocc2))
                Ht2ao[i0:i1] += _dot(eri, x2tmp).reshape(i1-i0,nao,nocc2).get()
    else:
        x2 = cupy.asarray(x2)
        Ht2ao = cupy.zeros(x2.shape)
        _dgemm = cupy.cuda.cublas.dgemm
        handle = cupy.cuda.device.get_cublas_handle()
        N = cupy.cuda.cublas.CUBLAS_OP_N
        T = cupy.cuda.cublas.CUBLAS_OP_T
        one = np.ones(1)
        one_ptr = one.ctypes.data
        x2_ptr = x2.data.ptr
        Ht2ao_ptr = Ht2ao.data.ptr
        def contract_vvvv_(eri, i0, i1, j0, j1):
            ic = i1 - i0
            jc = j1 - j0
            eri = eri.reshape(-1,jc*nao)
            #:Ht2[:,j0:j1] += np.einsum('xef,efab->xab', x2[:,i0:i1], eri)
            _dgemm(handle, N, N, jc*nao, nocc2, ic*nao,
                   one_ptr, eri.data.ptr, jc*nao, x2_ptr+i0*nao*8, nao2,
                   one_ptr, Ht2ao_ptr+j0*nao*8, nao2)

            if i0 > j0:
                #:Ht2[:,i0:i1] += np.einsum('xef,abef->xab', x2[:,j0:j1], eri)
                _dgemm(handle, T, N, ic*nao, nocc2, jc*nao,
                       one_ptr, eri.data.ptr, jc*nao, x2_ptr+j0*nao*8, nao2,
                       one_ptr, Ht2ao_ptr+i0*nao*8, nao2)

    l_ctr_offsets = vhfopt.l_ctr_offsets
    log_qs = vhfopt.log_qs
    ncptype = len(log_qs)
    cp_idx, cp_jdx = np.tril_indices(ncptype)

    if 1:
        eribuf = cupy.empty(blksize**2*nao**2)
        def fint(ish0, ish1, jsh0, jsh1, group_id):
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
            eri = cupy.ndarray((i1-i0, nao, j1-j0, nao), memptr=eribuf.data)
            eri[:] = 0.
            # strides to ensure data order consistent with eri(k1-k0,nao,l1-l0,nao)
            strides = [1, (j1-j0)*nao, (j1-j0)*nao**2, nao]
            ao_offsets = [0, 0, i0, j0]
            return _fill_eri_block(eri, strides, ao_offsets, vhfopt, group_id)
    else:
        intor = mol._add_suffix('int2e')
        ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                 'CVHFsetnr_direct_scf')
        eribuf = np.empty((blksize,blksize,nao,nao))
        loadbuf = np.empty((blksize,blksize,nao,nao))
        def fint(ish0, ish1, jsh0, jsh1, group_id):
            if ish0 != jsh0:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                eri = gto.moleintor.getints4c(
                    intor, mol._atm, mol._bas, mol._env,
                    shls_slice=(ish0,ish1,jsh0,jsh1), aosym='s2kl',
                    ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                aoblk = np.ndarray((i1-i0,nao,j1-j0,nao), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(aoblk.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, j0, j1),
                                       ctypes.c_int(nao))
            else:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                eri = gto.moleintor.getints4c(
                    intor, mol._atm, mol._bas, mol._env,
                    shls_slice=(ish0,ish1,ish0,ish1), aosym='s4',
                    ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                eri = lib.unpack_tril(eri, axis=0)
                aoblk = np.ndarray((i1-i0,nao,i1-i0,nao), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(aoblk.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, i0, i1),
                                       ctypes.c_int(nao))
            return cupy.asarray(aoblk)

    wVVoo = np.zeros((nao,nao,nocc,nocc))
    wVvoO = np.zeros((nao,nao,nocc,nocc))

    #mempool = cupy.get_default_memory_pool()
    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi,0]
        lj = vhfopt.uniq_l_ctr[cpj,0]
        if li > gpu_hf.LMAX_ON_GPU or lj > gpu_hf.LMAX_ON_GPU or log_q_ij.size == 0:
            continue

        ish0 = l_ctr_offsets[cpi]
        jsh0 = l_ctr_offsets[cpj]
        ish1 = l_ctr_offsets[cpi+1]
        jsh1 = l_ctr_offsets[cpj+1]
        aoblk = fint(ish0, ish1, jsh0, jsh1, cp_ij_id)

        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        contract_vvvv_(aoblk, i0, i1, j0, j1)

        #:fvv += 2*np.einsum('kc,kcab->ab', t1, eris_ovvv)
        #:fvv -= np.einsum('kc,kbca->ab', t1, eris_ovvv)
        # pppo = _einsum('PrQs,si->PrQi', aoblk, orbo)
        pppo = _dot(aoblk.reshape(-1,nao), orbo).reshape(i1-i0,nao,j1-j0,nocc)
        # tmp = _einsum('PrQi,Pj->rQij', pppo, t1po[i0:i1])
        tmp = _dot(pppo.reshape(i1-i0,-1).T, t1po[i0:i1]).reshape(nao,j1-j0,nocc,nocc)
        wVvoO[j0:j1] += tmp.get().transpose(1,0,2,3)
        tmp = None
        # pqji = _einsum('PrQi,rj->PQij', pppo, t1po)
        pqji = _dot(pppo.transpose(1,0,2,3).reshape(nao,-1).T,
                    t1po).reshape(i1-i0,j1-j0,nocc,nocc).get()
        wVVoo[i0:i1,j0:j1] = pqji
        pppo = pqji = None

        if ish0 != jsh0:
            wVVoo[j0:j1,i0:i1] = wVVoo[i0:i1,j0:j1].transpose(1,0,2,3)
            #mempool.free_all_blocks()
            ## TODO: optimize the transformation, too much mem used by reshape
            #tmp = _dot(aoblk.transpose(1,2,0,3).reshape(nao,-1).T, orbo)
            #wVvoO[i0:i1] += _dot(tmp.reshape(j1-j0,-1).T,
            #                     t1po[j0:j1]).reshape(i1-i0,nao,nocc,nocc).get()
            for i in range(i1 - i0):
                tmp = _dot(aoblk[i].reshape(nao,-1).T, orbo)
                wVvoO[i0+i] += _dot(tmp.reshape(j1-j0,-1).T,
                                    t1po[j0:j1]).reshape(nao,nocc,nocc).get()
        aoblk = tmp = None
    eribuf = loadbuf = tau = x2 = None

    if less_gpu_mem:
        Ht2ao = lib.transpose(Ht2ao.reshape(-1,nocc2)).reshape(nocc2,nao,nao)
    else:
        Ht2ao = Ht2ao.get()

    #:wVOov = lib.einsum('pqji,qb,pa->bjia', wVvoO, orbv, orbv)
    #:wVooV =-lib.einsum('pqji,pa,qb->bjia', wVVoo, orbv, orbv)
    tmp = np.einsum('pqii->pq', wVVoo)
    wpq = 2 * lib.einsum('pi,pq,qj->ij', vhfopt.coeff, tmp, vhfopt.coeff)
    wVVoo = cupy.asarray(wVVoo)
    qjia = _dot(wVVoo.reshape(nao,-1).T, -orbv)
    wVVoo = None
    wVooV = _dot(orbv.T, qjia.reshape(nao,-1)).reshape(nvir,nocc,nocc,nvir).get()
    qjia = None

    tmp = np.einsum('pqii->pq', wVvoO)
    wpq -= lib.einsum('pi,pq,qj->ji', vhfopt.coeff, tmp, vhfopt.coeff)
    wVvoO = cupy.asarray(wVvoO)
    qjia = _dot(wVvoO.reshape(nao,-1).T, orbv)
    wVvoO = None
    wVOov = _dot(orbv.T, qjia.reshape(nao,-1)).reshape(nvir,nocc,nocc,nvir).get()
    qjia = None

    #:t1new += 2*lib.einsum('edac,ikcd->ikea', eris_ovvv, t2)
    #:t1new +=  -lib.einsum('edac,ikdc->ikea', eris_ovvv, t2)
    Ht2full = ccsd._unpack_t2_tril(Ht2ao, nocc, nao, t2sym='jiba')
    Ht2full = Ht2full.reshape(nocc,nocc,nao,nao)
    orbo = mo[:,:nocc]
    orbv = mo[:,nocc:]
    t1tmp  = np.einsum('ijpq,qj->ip', Ht2full, orbo) * 2
    t1tmp -= np.einsum('ijqp,qj->ip', Ht2full, orbo)
    t1new = t1tmp.dot(orbv)
    Ht2full = t1tmp = None

    # vvvv-t2 contractions back to MO repr.
    Ht2tril = _ao2mo.nr_e2(Ht2ao, mo.conj(), (nocc,nmo,nocc,nmo), 's1', 's1')
    Ht2tril = Ht2tril.reshape(nocc2,nvir,nvir)

    # part of ovvv-t2 contractions back to MO repr.
    #: tmp = np.einsum('ijcd,ka,kdcb->ijba', tau, t1, eris.ovvv)
    #: t2new -= tmp + tmp.transpose(1,0,3,2)
    tmp = _ao2mo.nr_e2(Ht2ao, mo.conj(), (nocc,nmo,0,nocc), 's1', 's1')
    Ht2tril -= lib.ddot(tmp.reshape(nocc2*nvir,nocc), t1).reshape(nocc2,nvir,nvir)
    tmp = _ao2mo.nr_e2(Ht2ao, mo.conj(), (0,nocc,nocc,nmo), 's1', 's1')
    Ht2ao = None
    #: Ht2tril -= np.einsum('xkb,ka->xab', tmp.reshape(-1,nocc,nvir), t1)
    tmp = lib.transpose(tmp.reshape(nocc2,nocc,nvir), axes=(0,2,1))
    tmp = lib.ddot(tmp.reshape(nocc2*nvir,nocc), t1, 1,
                   np.ndarray((nocc2*nvir,nvir)), 0)
    tmp = lib.transpose(tmp.reshape(nocc2,nvir,nvir), axes=(0,2,1))
    Ht2tril -= tmp.reshape(nocc2,nvir,nvir)

    t2new = ccsd._unpack_t2_tril(Ht2tril, nocc, nvir, t2sym='jiba')

    if FREE_CUPY_CACHE:
        cupy.get_default_memory_pool().free_all_blocks()
    return wpq, t1new, t2new, wVOov, wVooV

def make_tau_tril(t1, t2):
    nocc, nvir = t1.shape
    nocc2 = nocc*(nocc+1)//2
    tau = np.empty((nocc2,nvir,nvir), dtype=t2.dtype)
    p1 = 0
    for i in range(nocc):
        p0, p1 = p1, p1 + i+1
        tau[p0:p1] = np.einsum('a,jb->jab', t1[i], t1[:i+1])
        tau[p0:p1] += t2[i,:i+1]
    return tau

def _fill_eri_block(eri, strides, ao_offsets, vhfopt, group_id):
    l_symb = lib.param.ANGULAR
    log_qs = vhfopt.log_qs
    nbins = 10
    ncptype = len(vhfopt.uniq_l_ctr)
    cp_idx, cp_jdx = np.tril_indices(ncptype)

    cp_kl_id = group_id
    log_q_kl = log_qs[group_id]
    if log_q_kl.size == 0:
        return eri

    nao = vhfopt.coeff.shape[0]
    cpk = cp_idx[group_id]
    cpl = cp_jdx[group_id]
    lk = vhfopt.uniq_l_ctr[cpk,0]
    ll = vhfopt.uniq_l_ctr[cpl,0]
    if lk > gpu_hf.LMAX_ON_GPU or ll > gpu_hf.LMAX_ON_GPU:
        raise NotImplementedError

    bins_locs_kl = gpu_hf._make_s_index_offsets(log_q_kl, nbins, vhfopt.direct_scf_tol)

    fn = libgint.GINTfill_int2e
    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi,0]
        lj = vhfopt.uniq_l_ctr[cpj,0]
        if li > gpu_hf.LMAX_ON_GPU or lj > gpu_hf.LMAX_ON_GPU or log_q_ij.size == 0:
            continue

        t0 = time.perf_counter()
        bins_locs_ij = gpu_hf._make_s_index_offsets(log_q_ij, nbins, vhfopt.direct_scf_tol)

        err = fn(vhfopt.bpcache,
                 ctypes.cast(eri.data.ptr, ctypes.c_void_p), ctypes.c_int(nao),
                 (ctypes.c_int*4)(*strides), (ctypes.c_int*4)(*ao_offsets),
                 bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                 bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(nbins), ctypes.c_int(cp_ij_id), ctypes.c_int(group_id))
        if err != 0:
            detail = f'CUDA Error for ({l_symb[li]}{l_symb[lj]}|{l_symb[lk]}{l_symb[ll]})'
            raise RuntimeError(detail)
        logger.debug1(vhfopt.mol, '(%s%s|%s%s) on GPU %.3fs',
                      l_symb[li], l_symb[lj], l_symb[lk], l_symb[ll],
                      time.perf_counter() - t0)
    return eri

def _make_eris_incore(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = ccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    # Cupy memory buffer may be created in previous SCF calculations.
    if FREE_CUPY_CACHE:
        cupy.get_default_memory_pool().free_all_blocks()

    mol = mycc.mol
    mo_coeff = np.asarray(eris.mo_coeff, order='F')
    nocc = eris.nocc
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc

    nao_cart = mycc.mol.nao_nr(cart=True)
    max_memory = max(MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    blksize = ((max_memory*.9e6-nocc**2*nao_cart**2*2*8)/8/nao_cart**2/2.5)**.5
    mem_avail = int(cupy.cuda.runtime.memGetInfo()[0] * .5)
    cupy.get_default_memory_pool().set_limit(mem_avail)
    blksize = max(BLKMIN, int(min((nao_cart+3)/4, blksize,
                                  (mem_avail*.5/8/nao_cart**2)**.5)))
    logger.debug1(mycc, 'blksize %d nao %d', blksize, nao_cart)

    vhfopt = gpu_hf._VHFOpt(mycc.mol, 'int2e')
    vhfopt.build(group_size=blksize, diag_block_with_triu=True)
    mol = vhfopt.mol
    mo = vhfopt.coeff.dot(mycc.mo_coeff)
    orbo = cupy.asarray(mo[:,:nocc])
    orbv = cupy.asarray(mo[:,nocc:])
    ao_loc = mol.ao_loc
    nao = mo.shape[0]

    l_ctr_offsets = vhfopt.l_ctr_offsets
    log_qs = vhfopt.log_qs
    ncptype = len(log_qs)
    cp_idx, cp_jdx = np.tril_indices(ncptype)

    ppOO = np.empty((nao,nao,nocc,nocc))
    pPoO = np.zeros((nao,nao,nocc,nocc))
    eribuf = cupy.empty(blksize**2*nao**2)
    #mempool = cupy.get_default_memory_pool()

    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi,0]
        lj = vhfopt.uniq_l_ctr[cpj,0]
        if li > gpu_hf.LMAX_ON_GPU or lj > gpu_hf.LMAX_ON_GPU or log_q_ij.size == 0:
            continue

        ish0 = l_ctr_offsets[cpi]
        jsh0 = l_ctr_offsets[cpj]
        ish1 = l_ctr_offsets[cpi+1]
        jsh1 = l_ctr_offsets[cpj+1]
        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        eri = cupy.ndarray((nao, i1-i0, j1-j0, nao), memptr=eribuf.data)
        eri[:] = 0.
        # strides to ensure data order consistent with eri(nao,k1-k0,l1-l0,nao)
        strides = [1, (i1-i0)*(j1-j0)*nao, (j1-j0)*nao, nao]
        ao_offsets = [0, 0, i0, j0]
        _fill_eri_block(eri, strides, ao_offsets, vhfopt, cp_ij_id)

        pijo = cupy.dot(eri.reshape(-1,nao), orbo)
        ijoo = cupy.dot(pijo.reshape(nao,-1).T, orbo)
        ppOO[i0:i1,j0:j1] = ijoo.get().reshape(i1-i0,j1-j0,nocc,nocc)
        ijoo = None

        jopi = cupy.asarray(pijo.reshape(nao*(i1-i0),(j1-j0)*nocc).T, order='C')
        jopo = cupy.dot(jopi.reshape(-1,i1-i0), orbo[i0:i1])
        pPoO[j0:j1] += jopo.get().reshape(j1-j0,nocc,nao,nocc).transpose(0,2,1,3)
        pijo = jopo = None

        if ish0 != jsh0:
            ppOO[j0:j1,i0:i1] = ppOO[i0:i1,j0:j1].transpose(1,0,2,3)
            opio = cupy.dot(jopi.reshape(j1-j0,-1).T, orbo[j0:j1])
            pPoO[i0:i1] += opio.get().reshape(nocc,nao,i1-i0,nocc).transpose(2,1,0,3)
            jopi = opio = None

    ppOO = cupy.asarray(ppOO)
    pooo = cupy.dot(ppOO.reshape(nao,-1).T, orbo)
    oooo = cupy.dot(pooo.reshape(nao,-1).T, orbo).reshape(nocc,nocc,nocc,nocc)
    ooov = cupy.dot(pooo.reshape(nao,-1).T, orbv).reshape(nocc,nocc,nocc,nvir)
    eris.oooo = oooo.get()
    eris.ovoo = lib.transpose(ooov.get().reshape(nocc*nocc,nocc*nvir)).reshape(nocc,nvir,nocc,nocc)
    pooo = oooo = ooov = None

    poov = cupy.dot(ppOO.reshape(nao,-1).T, orbv)
    oovv = cupy.dot(poov.reshape(nao,-1).T, orbv).reshape(nocc,nocc,nvir,nvir)
    eris.oovv = oovv.get()
    ppOO = poov = oovv = None

    pPoO = cupy.asarray(pPoO)
    poov = cupy.dot(pPoO.reshape(nao,-1).T, orbv)
    voov = cupy.dot(orbv.T, poov.reshape(nao,-1))
    eris.ovvo = lib.transpose(voov.get().reshape(nvir*nocc,nocc*nvir)).reshape(nocc,nvir,nvir,nocc)
    pPoO = poov = voov = None
    log.timer('CCSD integral transformation', *cput0)

    if FREE_CUPY_CACHE:
        cupy.get_default_memory_pool().free_all_blocks()
    return eris

class CCSD(ccsd.CCSD):
    device = 'gpu'
    update_amps = update_amps
    ao2mo = patch_cpu_kernel(ccsd.CCSD.ao2mo)(_make_eris_incore)


if __name__ == '__main__':
    from pyscf import scf, cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)],
    ]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    #mol.verbose = 4
    mol.build()
    mf = scf.RHF(mol)
    #mf = gpu_hf.RHF(mol)
    mf.run()
    mf.__dict__.update(scf.chkfile.load('h2o.chk', 'scf'))

    mcc = CCSD(mf)
    mcc.device = 'gpu'
    mcc.direct = True
    eris = mcc.ao2mo()
    print(lib.fp(eris.oooo) - -0.04187088041588999  )
    print(lib.fp(eris.ovoo) - -0.6596415803972335   )
    print(lib.fp(eris.oovv) - -0.8791061419180562   )
    print(lib.fp(eris.ovvo) - -0.0070595404496477925)
    #emp2, t1, t2 = mcc.init_amps(eris)
    #t1, t2 = update_amps(mcc, t1, t2, eris)
    t1 = np.load('t1.npy')
    t2 = np.load('t2.npy')
    t1, t2 = update_amps(mcc, t1, t2, eris)
    print(lib.fp(t1) - 0.00149855002713196187)
    print(lib.fp(t2) - 0.08513447607815229)
