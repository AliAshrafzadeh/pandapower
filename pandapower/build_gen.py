# -*- coding: utf-8 -*-

from __future__ import absolute_import
__author__ = 'tdess, lthurner, scheidler'

import sys
import numpy.core.numeric as ncn
import numpy as np
from pandapower.auxiliary import get_indices
from pypower.idx_gen import QMIN, QMAX, GEN_STATUS, GEN_BUS, PG, VG
from pypower.idx_bus import PV, REF, VA, VM, BUS_TYPE

def _build_gen_mpc(net, mpc, gen_is, eg_is, bus_lookup, enforce_q_lims, calculate_voltage_angles):
    '''
    Takes the empty mpc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The Pandapower format network

        **mpc** - The PYPOWER format network to fill in values
    '''
    eg_end = len(eg_is)
    gen_end = eg_end + len(gen_is)
    xw_end = gen_end + len(net["xward"])
    
    q_lim_default = 1e9 #which is 1000 TW - should be enough for distribution grids.

    # initialize generator matrix
    mpc["gen"] = np.zeros(shape=(xw_end, 21), dtype=float)
    mpc["gen"][:] = np.array([0, 0, 0, q_lim_default, -q_lim_default, 1., 1., 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # add ext grid / slack data
    mpc["gen"][:eg_end, GEN_BUS] = get_indices(eg_is["bus"].values, bus_lookup)
    mpc["gen"][:eg_end, VG] = eg_is["vm_pu"].values
    mpc["gen"][:eg_end, GEN_STATUS] = eg_is["in_service"].values
    
    #set bus values for external grid busses
    eg_busses = get_indices(eg_is["bus"].values, bus_lookup)
    if calculate_voltage_angles:
        mpc["bus"][eg_busses, VA] = eg_is["va_degree"].values
    mpc["bus"][eg_busses, BUS_TYPE] = REF
    
    # add generator / pv data
    if gen_end > eg_end:
        mpc["gen"][eg_end:gen_end, GEN_BUS] = get_indices(gen_is["bus"].values, bus_lookup)
        mpc["gen"][eg_end:gen_end, PG] = - gen_is["p_kw"].values * 1e-3 * gen_is["scaling"].values
        mpc["gen"][eg_end:gen_end, VG] = gen_is["vm_pu"].values
        
        #set bus values for generator busses
        gen_busses = get_indices(gen_is["bus"].values, bus_lookup)
        mpc["bus"][gen_busses, BUS_TYPE] = PV
        mpc["bus"][gen_busses, VM] = gen_is["vm_pu"].values

        
        if enforce_q_lims:
            mpc["gen"][eg_end:gen_end, QMIN] = -gen_is["max_q_kvar"].values * 1e-3
            mpc["gen"][eg_end:gen_end, QMAX] = -gen_is["min_q_kvar"].values * 1e-3
    
            qmax = mpc["gen"][eg_end:gen_end, [QMIN]]
            ncn.copyto(qmax, -q_lim_default, where=np.isnan(qmax))
            mpc["gen"][eg_end:gen_end, [QMIN]] = qmax
    
            qmin = mpc["gen"][eg_end:gen_end, [QMAX]]
            ncn.copyto(qmin, q_lim_default, where=np.isnan(qmin))
            mpc["gen"][eg_end:gen_end, [QMAX]] = qmin
    
    #add extended ward pv node data
    if xw_end > gen_end:
        xw = net["xward"]
        mpc["gen"][gen_end:xw_end, GEN_BUS] = get_indices(xw["ad_bus"].values, bus_lookup)
        mpc["gen"][gen_end:xw_end, VG] = xw["vm_pu"].values
        mpc["gen"][gen_end:xw_end, GEN_STATUS] = xw["in_service"]
        mpc["gen"][gen_end:xw_end, QMIN] = -q_lim_default
        mpc["gen"][gen_end:xw_end, QMAX] = q_lim_default
        
        xward_busses = get_indices(net["xward"]["ad_bus"].values, bus_lookup)
        mpc["bus"][xward_busses, BUS_TYPE] = PV
        mpc["bus"][xward_busses, VM] = net["xward"]["vm_pu"].values

          
    