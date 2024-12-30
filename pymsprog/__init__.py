
"""
This module defines the functions to compute MS progression.
"""

__version__ = "0.1.1"


import numpy as np
import pandas as pd

import datetime

import copy

from inspect import signature, Parameter

import warnings

#####################################################################################

def MSprog(data, subj_col, value_col, date_col, outcome, subjects=None,
           validconf_col=None, # tmp - for opera
           relapse=None, rsubj_col=None, rdate_col=None,
           delta_fun=None, worsening=None, valuedelta_col=None,
           event='firstprog', baseline='fixed',
           conf_weeks=12, conf_tol_days=30, conf_unbounded_right=False,
           require_sust_weeks=0, check_intermediate=True,
           relapse_to_bl=30, relapse_to_event=0, relapse_to_conf=30, relapse_assoc=90, relapse_indep=None,
           sub_threshold=False, bl_geq=False, relapse_rebl=False, min_value=None, prog_last_visit=False,
           include_dates=False, include_value=False, include_stable=True, verbose=1):
    """
    Compute MS progression from longitudinal data.
    ARGUMENTS:
        data, DataFrame: longitudinal data containing subject ID, outcome value, date of visit
        subj_col, str: name of data column with subject ID
        value_col, str: name of data column with outcome value
        date_col, str: name of data column with date of visit
        outcome, str: outcome type. Must be one of the following:
                        - 'edss' [Expanded Disability Status Scale]
                        - 'nhpt' [Nine-Hole Peg Test]
                        - 't25fw' [Timed 25-Foot Walk]
                        - 'sdmt' [Symbol Digit Modalities Test]
                        - None [only accepted when specifying a custom delta_fun]
        subjects, list-like : (optional) subset of subjects
        relapse, DataFrame: (optional) longitudinal data containing subject ID and relapse date
        rsubj_col / rdate_col, str: name of columns for relapse data, if different from outcome data
        delta_fun, function: (optional) Custom function specifying the minimum delta corresponding to a valid change from baseline.
        worsening, str: direction of worsening ('increase' or 'decrease'); automatically set to 'increase' for edss, nhpt, t25fw, to 'decrease' for sdmt.
        event, str: 'first' [only the very first event - improvement or progression]
                    'firsteach' [first improvement and first progression]
                    'firstprog' [first progression]
                    'firstprogtype' [first progression of each kind - PIRA, RAW, undefined]
                    'firstPIRA' [first PIRA]
                    'firstRAW' [first RAW]
                    'multiple'[all events, default]
        baseline, str: 'fixed'[default], 'roving', 'roving_impr', 'roving_wors'
        conf_weeks, int or list-like : period before confirmation (weeks)
        conf_tol_days, int or list-like of length 1 or 2: tolerance window for confirmation visit (days): [t(weeks)-conf_tol_days[0](days), t(weeks)+conf_tol_days[0](days)]
        conf_unbounded_right, bool: if True, confirmation window is [t(weeks)-conf_tol_days, inf)
        require_sust_weeks, int: count an event as such only if sustained for _ weeks from confirmation
        relapse_to_bl, int: minimum distance from a relapse (days) for a visit to be used as baseline (otherwise the next available visit is used as baseline).
        relapse_to_event, int: minimum distance from a relapse (days) for an event to be considered as such.
        relapse_to_conf, int: minimum distance from a relapse (days) for a visit to be a valid confirmation visit.
        relapse_assoc, int: maximum distance from last relapse for a progression event to be considered as RAW (days)
        relapse_indep, dict: relapse-free intervals around baseline, event, and confirmation to define PIRA.
                            {'bl':(b0,b1), 'event':(e0,e1), 'conf':(c0,c1)}
                            If the right end is None, the interval is assumed to extend up to the left end of the next interval.
                            If the left end is None, the interval is assumed to extend up to the right end of the previous interval.
                            Examples:
                                - {'bl':(0,None), 'event':(None, None), 'conf':(None,0)} [default]
                                    no relapses between baseline and confirmation (high-specificity def), Muller JAMA Neurol 2023
                                - {'bl':(0,None), 'event':(None, 30), 'conf':(30,30)}
                                    no relapses within baseline->event+30dd and within confirmation+-30dd, Kappos JAMA Neurol 2020
                                - {'bl':(0,0), 'event':(90, 30), 'conf':(90, 30)}
                                    no relapses within event-90dd->event+30dd and within confirmation-90dd->confirmation+30dd, Muller JAMA Neurol 2023
        sub_threshold, bool: if True, include confirmed sub-threshold events for roving baseline
        bl_geq, bool: if True, new reference must always be >= than previous reference. When it's not, the old value is assigned to it.
        relapse_rebl, bool: if True, re-baseline after every relapse
        min_value, float: only consider progressions events where the outcome is >= value (default is None, i.e., no threshold)
        prog_last_visit, bool: if True, include progressions occurring at last visit (i.e. with no confirmation)
        include_dates, bool: if True, report dates of events
        include_value, bool: if True, report value of outcome at event
        include_stable, bool: if True, include subjects with no events in extended info
        verbose, int: 0[print no info], 1[print concise info], 2[default, print extended info]
    RETURNS:
        Two DataFrames:
         - summary of detected events for each subject;
         - extended info on event sequence for each subject.
    """

    #####################################################################################
    # SETUP

    #### for debugging
    #np.seterr(all='raise')
    ####

    warning_msgs = []

    if outcome is None or outcome.lower() not in ['edss', 'nhpt', 't25fw', 'sdmt']:
        outcome = None
    else:
        outcome = outcome.lower()

    if min_value is None:
        min_value = - np.inf

    if prog_last_visit==True:
        prog_last_visit = np.inf

    try:
       _ = (e for e in conf_weeks) # check if conf_weeks is iterable
    except TypeError:
       conf_weeks = [conf_weeks] # if it's not, make it a list with a single element

    try:
        _ = (e for e in conf_tol_days) # check if conf_tol_days is iterable
    except TypeError:
        conf_tol_days = [conf_tol_days, conf_tol_days] # if it's not, make it a pair with equal elements

    data = data.copy()

    ######( #_rrebl_#
    # if event=='firstRAW':
    #     relapse_rebl = False
    ######)

    if rsubj_col is None:
        rsubj_col = subj_col
    if rdate_col is None:
        rdate_col = date_col
    if validconf_col is None:
        validconf_col = 'validconf'
        data.loc[:, validconf_col] = 1
    if valuedelta_col is None:
        valuedelta_col = value_col

    if relapse_indep is None:
        relapse_indep = {'bl': (0, 0), 'event': (90, 30), 'conf': (90, 30)}


    if outcome in ('edss', 'nhpt', 't25fw'):
        worsening = 'increase'
    elif outcome=='sdmt':
        worsening = 'decrease'
    elif worsening is None:
        raise ValueError('Either specify an outcome type, or specify the direction of worsening (\'increase\' or \'decrease\')')
    #_c_########
    # def delta(value):
    #     return compute_delta(value, outcome) if delta_fun is None else delta_fun(value)

    # Define local `is_event` function
    def isevent_loc(x, baseline, type='prog', st=False, baseline_delta=None):
        return is_event(x, baseline, type=type, outcome=outcome, worsening=worsening,
                        sub_threshold=st, delta_fun=delta_fun, baseline_delta=baseline_delta)
    #_c_########


    # Remove missing values from columns of interest
    data = data[np.unique([subj_col, value_col, date_col, validconf_col, valuedelta_col])].copy().dropna()  #_c_#

    # Convert dates to datetime.date format
    data[date_col] = pd.to_datetime(data[date_col]) #col_to_date(data[date_col]) #
    if relapse is None:
        relapsedata = False
        relapse_rebl = False
        relapse = pd.DataFrame([], columns=[rsubj_col, rdate_col])
        relapse_start = data[date_col].min()
    else:
        relapsedata = True
        relapse = relapse[[rsubj_col, rdate_col]].copy().dropna()
        relapse[rdate_col] = pd.to_datetime(relapse[rdate_col]) #col_to_date(relapse[rdate_col]) #
        relapse_start = relapse[rdate_col].min()


    # Convert dates to days from minimum #_d_#
    global_start = min(data[date_col].min(), relapse_start)
    if relapsedata:
        relapse[rdate_col] = (relapse[rdate_col] - global_start).apply(lambda x : x.days)
    else:
        relapse[rdate_col] = relapse[rdate_col].astype(int)
    data[date_col] = (data[date_col] - global_start).apply(lambda x : x.days)

    if subjects is not None:
        data = data[data[subj_col].isin(subjects)]
        relapse = relapse[relapse[rsubj_col].isin(subjects)]

    # Check if values are in correct range
    if outcome is not None:
        for vcol in np.unique([value_col, valuedelta_col]):
            if (data[vcol]<0).any():
                raise ValueError('invalid %s scores' %outcome.upper())
            elif outcome=='edss' and (data[vcol]>10).any():
                raise ValueError('invalid %s scores' %outcome.upper())
            elif outcome=='sdmt' and (data[vcol]>110).any():
                raise ValueError('SDMT scores >110')
            elif outcome=='nhpt' and (data[vcol]>300).any():
                warning_msgs.append('NHPT scores >300')
            elif outcome=='t25fw' and (data[vcol]>180).any():
                warning_msgs.append('T25FW scores >180')

    #####################################################################################
    # Assess progression

    all_subj = data[subj_col].unique()
    nsub = len(all_subj)
    max_nevents = round(data.groupby(subj_col)[date_col].count().max()/2)
    results = pd.DataFrame([[None]*10 + [None]*len(conf_weeks)
                            + [None]*(len(conf_weeks)) + [None]*2]*nsub*max_nevents, #*(len(conf_weeks)-1) #_piraconf_#
               columns=[subj_col, 'nevent', 'event_type', 'bldate', 'blvalue', 'date', 'value',
                        'total_fu', 'time2event', 'bl2event']
                       + ['conf'+str(m) for m in conf_weeks]+ ['PIRA_conf'+str(m) for m in conf_weeks] #[1:]]  #_piraconf_#
                       + ['sust_days', 'sust_last'])
    results[subj_col] = np.repeat(all_subj, max_nevents)
    results['nevent'] = np.tile(np.arange(1,max_nevents+1), nsub)
    summary = pd.DataFrame([['']+[0]*5]*nsub, columns=['event_sequence', 'improvement', 'progression',
                                                  'RAW', 'PIRA', 'undefined_prog'], index=all_subj)
    total_fu = {s : 0 for s in all_subj}

    for subjid in all_subj:

        data_id = data.loc[data[subj_col]==subjid,:].copy()

        # If more than one visit occur on the same day, only keep last
        udates, ucounts = np.unique(data_id[date_col].values, return_counts=True)
        if any(ucounts>1):
            data_id = data_id.groupby(date_col).last().reset_index()
            # groupby() indexes the dataframe by date_col: resetting index to convert date_col back into a normal column

        # Sort visits in chronological order
        sorted_tmp = data_id.sort_values(by=[date_col])
        if any(sorted_tmp.index != data_id.index):
            raise TypeError('uffa')
            data_id = sorted_tmp.copy()

        nvisits = len(data_id)
        first_visit = data_id[date_col].min()
        relapse_id = relapse.loc[relapse[rsubj_col]==subjid,:].copy().reset_index(drop=True)
        relapse_id = relapse_id.loc[relapse_id[rdate_col] >= first_visit - relapse_to_bl,:] # ignore relapses occurring before first visit
                                            #_d_# first_visit-datetime.timedelta(days=relapse_to_bl)
        relapse_dates = relapse_id[rdate_col].values
        nrel = len(relapse_dates)

        if verbose == 2:
            print('\nSubject #%s: %d visit%s, %d relapse%s'
              %(subjid, nvisits,'' if nvisits==1 else 's', nrel, '' if nrel==1 else 's'))
            if any(ucounts>1):
                print('Found multiple visits on the same day: only keeping last.')
            if any(sorted_tmp.index != data_id.index):
                print('Visits not listed in chronological order: sorting them.')

        data_id.reset_index(inplace=True, drop=True)

        total_fu[subjid] = data_id.loc[nvisits-1,date_col] - data_id.loc[0,date_col]

        #_d_#
        # all_dates, sorted_ind = np.unique(list(data_id[date_col]) + list(relapse_dates), #np.concatenate([data_id[date_col].values, relapse_dates]),
        #                       return_index=True) # numpy unique() returns sorted values
        # is_rel = [x in relapse_dates for x in all_dates] # whether a date corresponds to a relapse
        # # If there is a relapse with no visit, readjust the indices:
        # date_dict = {sorted_ind[i] : i for i in range(len(sorted_ind))}

        relapse_df = pd.DataFrame([relapse_dates]*len(data_id))
        relapse_df['visit'] = data_id[date_col].values
        dist = relapse_df.drop(['visit'],axis=1).subtract(relapse_df['visit'], axis=0) #_d_# #.apply(lambda x : pd.to_timedelta(x).dt.days)
        distm = - dist.mask(dist>0)  # other=-float('inf')
        distp = dist.mask(dist<0)  # other=float('inf')
        distm[distm.isna()] = float('inf')
        distp[distp.isna()] = float('inf')
        data_id['closest_rel-'] = float('inf') if all(distm.isna()) else distm.min(axis=1)
        data_id['closest_rel+'] = float('inf') if all(distp.isna()) else distp.min(axis=1)

        event_type, event_index = [''], []
        bldate, blvalue, edate, evalue, time2event, bl2event = [], [], [], [], [], []
        conf, sustd, sustl = {m : [] for m in conf_weeks}, [], []
        pira_conf = {m : [] for m in conf_weeks} #[1:]}  #_piraconf_#


        bl_idx, search_idx = 0, 1 # baseline index and index of where we are in the search
        proceed = 1
        ##### #_rrebl_# (
        # phase = 0 # if post-relapse re-baseline is enabled (relapse_rebl==True),
        #           # phase will become 1 when re-searching for PIRA events
        ##### )
        conf_window = [(int(c*7) - conf_tol_days[0], float('inf')) if conf_unbounded_right
                       else (int(c*7) - conf_tol_days[0], int(c*7) + conf_tol_days[1]) for c in conf_weeks]
        irel = 0 if nrel==0 else next((r for r in range(nrel) if relapse_dates[r] > data_id.loc[bl_idx, date_col]), None)
        bl_last = None

        while proceed:

            # Set baseline (skip if within relapse influence)
            while proceed and data_id.loc[bl_idx,'closest_rel-'] < relapse_to_bl:
                if verbose==2:
                    print('Baseline (visit no.%d) is within relapse influence: moved to visit no.%d'
                              %(bl_idx+1, bl_idx+2))
                bl_idx += 1
                search_idx += 1
                if bl_idx > nvisits-2:
                    proceed = 0
                    if verbose == 2:
                        print('Not enough visits left: end process')

            if bl_idx > nvisits - 1:
                bl_idx = nvisits - 1
                proceed = 0
                if verbose == 2:
                    print('Not enough visits left: end process')
            elif bl_geq and bl_last is not None and bl_last > data_id.loc[bl_idx, value_col]:
                ########## Kappos2020 (by Sean Yiu)
                data_id.loc[bl_idx, value_col] = bl_last
                #########

            bl = data_id.iloc[bl_idx,:]
            bl_last = bl[value_col]

            # Event detection
            change_idx = next((x for x in range(search_idx, nvisits)
                    if isevent_loc(data_id.loc[x,value_col], bl[value_col], type='change',
                        st=sub_threshold, baseline_delta=data_id.loc[bl_idx,valuedelta_col]) # first occurring value!=baseline
                        and (data_id.loc[x, 'closest_rel-'] >= relapse_to_event)), None) # occurring at least `relapse_to_event` days from last relapse
            #_c_# data_id.loc[x,value_col]!=bl[value_col]
            if change_idx is None: # value does not change in any subsequent visit
                conf_idx = []
                conf_t = {}
                proceed = 0
                if verbose == 2:
                    print('No %s change in any subsequent visit: end process' %outcome.upper())
            elif (relapse_rebl and irel is not None
                  and nrel > irel and change_idx + (0 if event=='firstPIRA' else relapse_assoc) >= relapse_dates[irel]
                ):
                search_idx = change_idx
            else:
                ###### #_conf_#
                # conf_idx = [next((x for x in range(change_idx+1, nvisits)
                #         if c[0] <= data_id.loc[x,date_col] - data_id.loc[change_idx,date_col] <= c[1] # date in confirmation range
                #         and data_id.loc[x,'closest_rel-'] >= relapse_to_conf), # occurring at least `relapse_to_conf` days from last relapse
                #         None) for c in conf_window]
                # conf_t = [conf_weeks[i] for i in range(len(conf_weeks)) if conf_idx[i] is not None]
                # conf_idx = [ic for ic in conf_idx if ic is not None]
                ###### #_conf_#
                conf_idx = [[x for x in range(change_idx+1, nvisits)
                        if c[0] <= data_id.loc[x,date_col] - data_id.loc[change_idx,date_col] <= c[1] # date in confirmation range
                        and data_id.loc[x,'closest_rel-'] >= relapse_to_conf # occurring at least `relapse_to_conf` days from last relapse
                        and data_id.loc[x, validconf_col]] # can be used as confirmation
                        for c in conf_window]
                conf_t = {conf_weeks[i] : conf_idx[i] for i in range(len(conf_weeks))}
                conf_idx = np.unique([x for i in range(len(conf_idx)) for x in conf_idx[i]])
                ###### #_conf_#
                if verbose == 2:
                    print('%s change at visit no.%d (%s); potential confirmation visits available: no.%s'
                          %(outcome.upper(), change_idx+1 ,
                            global_start + datetime.timedelta(days=data_id.loc[change_idx,date_col].item()), #_d_#
                            [i+1 for i in conf_idx]))

                # Confirmation
                # ============

                # CONFIRMED IMPROVEMENT:
                # --------------------
                if (len(conf_idx) > 0 # confirmation visits available
                        and isevent_loc(data_id.loc[change_idx,value_col], bl[value_col], type='impr',
                                        baseline_delta=data_id.loc[bl_idx,valuedelta_col]) # value decreased (>delta) from baseline
                        and (all([isevent_loc(data_id.loc[x,value_col], bl[value_col], type='impr',
                                              baseline_delta=data_id.loc[bl_idx,valuedelta_col])
                                 for x in range(change_idx+1,conf_idx[0]+1)]) # decrease is confirmed at all visits between event and confirmation visit
                            if check_intermediate else isevent_loc(data_id.loc[conf_idx[0],value_col], bl[value_col],
                                                        type='impr', baseline_delta=data_id.loc[bl_idx,valuedelta_col]))
                        ######( #_rrebl_#
                        #and phase == 0 # skip if re-checking for PIRA after post-relapse re-baseline
                        ######)
                        # and not ((event in ('firstprog', 'firstprogtype', 'firstPIRA', 'firstRAW'))
                        #          and baseline == 'fixed')  # skip this event if only searching for progressions with a fixed baseline
                    ):
                    next_change = next((x for x in range(conf_idx[0]+1,nvisits)
                        if not isevent_loc(data_id.loc[x,value_col], bl[value_col], type='impr',
                                           baseline_delta=data_id.loc[bl_idx,valuedelta_col])), None) #_c_# data_id.loc[x,value_col] - bl[value_col] > - delta(bl[value_col])
                    conf_idx = conf_idx if next_change is None else [ic for ic in conf_idx if ic<next_change] # confirmed visits
                    #_conf_# #conf_t = conf_t[:len(conf_idx)]
                    # sustained until:
                    next_nonsust = next((x for x in range(conf_idx[0]+1,nvisits) #_r_# #conf_idx[-1]
                    if not isevent_loc(data_id.loc[x,value_col], bl[value_col], type='impr',
                                       baseline_delta=data_id.loc[bl_idx,valuedelta_col]) #_c_# # decrease not sustained
                        ), None)

                    valid_impr = 1
                    if require_sust_weeks:
                        if not check_intermediate and ((data_id.loc[nvisits-1,date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_weeks*7):
                            sust_vis = next((x for x in range(change_idx+1, nvisits) if (data_id.loc[x,date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_weeks*7))
                        else:
                            sust_vis = nvisits - 1
                        valid_impr = ((next_nonsust is None) or (data_id.loc[next_nonsust,date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_weeks*7 #.days #_d_# # improvement sustained up to end of follow-up, or for `require_sust_weeks` weeks
                                      ) if check_intermediate else isevent_loc(data_id.loc[sust_vis,value_col], # improvement confirmed at last visit, or first visit after `require_sust_weeks` weeks
                                    bl[value_col], type='impr', baseline_delta=data_id.loc[bl_idx,valuedelta_col])
                    if valid_impr:
                        sust_idx = nvisits-1 if next_nonsust is None else next_nonsust-1

                        event_type.append('impr')
                        event_index.append(change_idx)
                        bldate.append(global_start + datetime.timedelta(days=bl[date_col].item())) #_d_#
                        blvalue.append(bl[value_col])
                        edate.append(global_start + datetime.timedelta(days=data_id.loc[change_idx,date_col].item())) #_d_#
                        evalue.append(data_id.loc[change_idx,value_col])
                        time2event.append(data_id.loc[change_idx,date_col] - data_id.loc[0,date_col]) #.days #_d_#
                        bl2event.append(data_id.loc[change_idx,date_col] - bl[date_col]) #.days #_d_#
                        for m in conf_weeks:
                            confirmed_at = np.intersect1d(conf_t[m], conf_idx)
                            if len(confirmed_at)==0:
                                del conf_t[m]
                            conf[m].append(1 if len(confirmed_at)>0 else 0) #_conf_# 1 if m in conf_t else 0
                        for m in conf_weeks: #[1:]: #_piraconf_#
                            pira_conf[m].append(None)
                        sustd.append(data_id.loc[sust_idx,date_col] - data_id.loc[change_idx,date_col]) #.days #_d_#
                        sustl.append(int(sust_idx == nvisits-1)) #int(data_id.loc[nvisits-1,value_col] - bl[value_col] <= - delta(bl[value_col]))


                        # Only keep first available confirmation visit for each value in conf_weeks
                        conf_idx = [np.min([x for x in conf_t[m] if x in conf_idx]) for m in conf_t.keys()] #_conf_#

                        # next change from first confirmation
                        next_change = next((x for x in range(conf_idx[0]+1,nvisits) #_r_# #conf_idx[-1]
                        if not isevent_loc(data_id.loc[x,value_col], bl[value_col], type='impr',
                                           baseline_delta=data_id.loc[bl_idx,valuedelta_col]) #_c_# # either decrease not sustained
                        or isevent_loc(data_id.loc[x,value_col], data_id.loc[conf_idx[0], value_col], #_r_# #conf_idx[-1]
                                       type='change', baseline_delta=data_id.loc[bl_idx,valuedelta_col]) # or further valid change from confirmation
                                    ), None)

                        if baseline in ('roving', 'roving_impr'):
                            bl_idx = conf_idx[0] # set new baseline at first confirmation time
                            search_idx = nvisits if next_change is None else next_change
                        else:
                            search_idx = nvisits if next_change is None else next_change #next_nonsust

                        if verbose == 2:
                            print('%s improvement (visit no.%d, %s) confirmed at %s weeks, sustained up to visit no.%d (%s)'
                                  %(outcome.upper(), change_idx+1,
                                    global_start + datetime.timedelta(days=data_id.loc[change_idx,date_col].item()), #_d_#
                                    ', '.join([str(x) for x in conf_t.keys()]),  #_conf_#
                                    sust_idx+1,
                                    global_start + datetime.timedelta(days=data_id.loc[sust_idx,date_col].item()))) #_d_#
                            print('Baseline at visit no.%d, searching for events from visit no.%s on'
                                  %(bl_idx+1, '-' if search_idx>=nvisits else search_idx+1))

                    else:
                        search_idx = change_idx + 1 # skip the change and look for other patterns after it
                        if verbose == 2:
                            print('Change confirmed but not sustained for >=%d weeks: proceed with search'
                                  %require_sust_weeks)

                # Confirmed sub-threshold improvement: RE-BASELINE
                # ------------------------------------------------
                elif (len(conf_idx) > 0 # confirmation visits available
                        and data_id.loc[change_idx,value_col]<bl[value_col] # value decreased from baseline
                        and (all([data_id.loc[x,value_col]<bl[value_col]
                                 for x in range(change_idx+1,conf_idx[0]+1)])  # decrease is confirmed
                        if check_intermediate else data_id.loc[conf_idx[0],value_col]<bl[value_col])
                        and baseline in ('roving', 'roving_impr') and sub_threshold
                        ######( #_rrebl_#
                        # and phase == 0 # skip if re-checking for PIRA after post-relapse re-baseline
                        ######)
                        ):
                    next_change = next((x for x in range(conf_idx[0]+1,nvisits)
                        if data_id.loc[x,value_col] >= bl[value_col]), None)
                    bl_idx = conf_idx[0] # set new baseline at first confirmation time
                    search_idx = nvisits if next_change is None else next_change
                    if verbose == 2:
                        print('Confirmed sub-threshold %s improvement (visit no.%d)'
                              %(outcome.upper(), change_idx+1))
                        print('Baseline at visit no.%d, searching for events from visit no.%s on'
                              %(bl_idx+1, '-' if search_idx is None else search_idx+1))

                # CONFIRMED PROGRESSION:
                # ---------------------
                elif (data_id.loc[change_idx,value_col] >= min_value
                        and isevent_loc(data_id.loc[change_idx,value_col], bl[value_col], type='prog',
                                        baseline_delta=data_id.loc[bl_idx,valuedelta_col]) #_c_# # value increased (>delta) from baseline
                    and ((len(conf_idx) > 0 # confirmation visits available
                        and (all([isevent_loc(data_id.loc[x,value_col], bl[value_col], type='prog',
                                              baseline_delta=data_id.loc[bl_idx,valuedelta_col]) #_c_#
                                 for x in range(change_idx+1,conf_idx[0]+1)]) # increase is confirmed at (all visits up to) first valid date
                            if check_intermediate else isevent_loc(data_id.loc[conf_idx[0],value_col], bl[value_col],
                                                        type='prog', baseline_delta=data_id.loc[bl_idx,valuedelta_col]))
                        and all([data_id.loc[x,value_col] >= min_value for x in range(change_idx+1,conf_idx[0]+1)]) # confirmation above min_value too
                        ) or (data_id.loc[change_idx, date_col] - data_id.loc[0, date_col] <= prog_last_visit*7
                              and change_idx==nvisits-1))
                      ):
                    if change_idx==nvisits-1:
                        conf_idx = [nvisits-1]
                    next_change = next((x for x in range(conf_idx[0]+1,nvisits)
                        if not isevent_loc(data_id.loc[x,value_col], bl[value_col], type='prog',
                                           baseline_delta=data_id.loc[bl_idx,valuedelta_col])), None)  #_c_#
                    conf_idx = conf_idx if next_change is None else [ic for ic in conf_idx if ic<next_change] # confirmed dates
                    #_conf_# #conf_t = conf_t[:len(conf_idx)]
                    # sustained until:
                    next_nonsust = next((x for x in range(conf_idx[0]+1,nvisits) #_r_# #conf_idx[-1]
                        if not isevent_loc(data_id.loc[x,value_col], bl[value_col], type='prog',
                                           baseline_delta=data_id.loc[bl_idx,valuedelta_col]) #_c_# # increase not sustained
                                    ), None)
                    valid_prog = 1
                    if require_sust_weeks:
                        if not check_intermediate and ((data_id.loc[nvisits-1,date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_weeks*7):
                            sust_vis = next((x for x in range(change_idx+1, nvisits) if (data_id.loc[x,date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_weeks*7))
                        else:
                            sust_vis = nvisits - 1
                        valid_prog = ((next_nonsust is None) or (data_id.loc[next_nonsust,date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_weeks*7 #.days #_d_# # progression sustained up to end of follow-up, or for `require_sust_weeks` weeks
                                      ) if check_intermediate else isevent_loc(data_id.loc[sust_vis,value_col], # progression confirmed at last visit, or first visit after `require_sust_weeks` weeks
                                    bl[value_col], type='prog', baseline_delta=data_id.loc[bl_idx,valuedelta_col])
                    if valid_prog:

                        include_event = True

                        nev = len(event_type)

                        sust_idx = nvisits-1 if next_nonsust is None else next_nonsust-1

                        if (data_id.loc[change_idx,'closest_rel-'] <= relapse_assoc # event is relapse-associated
                            ######( #_rrebl_#
                            # and phase==0
                            ######)
                            ):
                            if event=='firstPIRA' and baseline=='fixed':
                                # skip this event if searching for PIRA only (with a fixed baseline)
                                if verbose==2:
                                    print('Worsening confirmed but not a PIRA event: skipped')
                                include_event = False
                            else:
                                event_type.append('RAW')
                                event_index.append(change_idx)
                        elif data_id.loc[change_idx,'closest_rel-'] > relapse_assoc: # event is not relapse-associated
                            if (event=='firstRAW' and baseline=='fixed'
                            ######( #_rrebl_#
                            # and phase==0
                            ######),
                            ):
                                # skip this event if only searching for RAW with a fixed baseline
                                if verbose == 2:
                                    print('Worsening confirmed but not a RAW event: skipped')
                                include_event = False
                            else:
                                # Check if it's PIRA *(
                                intervals = {ic : [] for ic in conf_idx}
                                for ic in conf_idx:
                                    for point in ('bl', 'event', 'conf'):
                                        t = bl[date_col] if point=='bl' else (data_id.loc[change_idx,date_col]
                                                if point=='event' else data_id.loc[ic,date_col])
                                        if relapse_indep[point][0] is not None:
                                            t0 = t - relapse_indep[point][0]
                                        if relapse_indep[point][1] is not None:
                                            t1 = t + relapse_indep[point][1]
                                            if t1>t0:
                                                intervals[ic].append([t0,t1])
                                rel_inbetween = [np.logical_or.reduce([(a[0]<=relapse_dates) & (relapse_dates<=a[1])
                                                for a in intervals[ic]]).any() for ic in conf_idx]

                                pconf_idx = [conf_idx[i] for i in range(len(conf_idx)) if not rel_inbetween[i]]  #_piraconf_#
                                # pconf_idx = conf_idx if not any(rel_inbetween) else conf_idx[:next(i for i in
                                #                                         range(len(conf_idx)) if rel_inbetween[i])]
                                pconf_t = copy.deepcopy(conf_t) #_conf_# [conf_t[i] for i in range(len(conf_t)) if not rel_inbetween[i]] #conf_t[:len(pconf_idx)] # #_piraconf_#
                                if len(pconf_idx)>0: # if pira is confirmed
                                    ######## #_conf_#
                                    for m in conf_weeks: #[1:]: #_piraconf_#
                                        confirmed_at = np.intersect1d(pconf_t[m], pconf_idx)
                                        if len(confirmed_at)==0:
                                            del pconf_t[m]
                                        pira_conf[m].append(int(len(confirmed_at)>0)) #_conf_# int(m in conf_t)
                                    ######## #_conf_#

                                    event_type.append('PIRA')
                                    event_index.append(change_idx)
                                elif event=='firstPIRA' and baseline=='fixed': # #_rrebl_# elif phase == 0: # if pira is not confirmed, and we're not re-searching for pira events only
                                    if verbose==2:
                                        print('Worsening confirmed but not a PIRA event: skipped')
                                    include_event = False
                                else:
                                    event_type.append('prog')
                                    event_index.append(change_idx)
                                # )*

                        if include_event:
                            # **(
                            if event_type[-1] != 'PIRA': # #_rrebl_# and phase==0
                                for m in conf_weeks: #[1:]: #_piraconf_#
                                    pira_conf[m].append(None)

                            # #_rrebl_# if event_type[-1] == 'PIRA' or phase == 0:
                            bldate.append(global_start + datetime.timedelta(days=bl[date_col].item())) #_d_#
                            blvalue.append(bl[value_col])
                            edate.append(global_start + datetime.timedelta(days=data_id.loc[change_idx,date_col].item())) #_d_#
                            evalue.append(data_id.loc[change_idx,value_col])
                            time2event.append(data_id.loc[change_idx,date_col] - data_id.loc[0,date_col]) #.days #_d_#
                            bl2event.append(data_id.loc[change_idx,date_col] - bl[date_col]) #.days #_d_#
                            for m in conf_weeks:
                                confirmed_at = np.intersect1d(conf_t[m], conf_idx)
                                if len(confirmed_at)==0:
                                    del conf_t[m]
                                conf[m].append(1 if len(confirmed_at)>0 else 0) #_conf_# 1 if m in conf_t else 0
                            sustd.append(data_id.loc[sust_idx,date_col] - data_id.loc[change_idx,date_col]) #.days #_d_#
                            sustl.append(int(sust_idx == nvisits-1))
                            if verbose == 2:
                                print('%s progression[%s] (visit no.%d, %s) confirmed at %s weeks, sustained up to visit no.%d (%s)'
                                      %(outcome.upper(), event_type[-1], change_idx+1,
                                        global_start + datetime.timedelta(days=data_id.loc[change_idx,date_col].item()), #_d_#
                                        ', '.join([str(x) for x in (pconf_t.keys() if event_type[-1]=='PIRA'
                                                                    else conf_t.keys())]), #_conf_#
                                        sust_idx+1,
                                        global_start + datetime.timedelta(days=data_id.loc[sust_idx,date_col].item()))) #_d_#
                            ######( #_rrebl_#
                            # else:
                            #     for m in conf_weeks:
                            #         confirmed_at = np.intersect1d(conf_t[m], conf_idx)
                            #         if len(confirmed_at)==0:
                            #             del conf_t[m]
                            ######)
                            # )**

                    else:
                        if verbose == 2:
                            print('Change confirmed but not sustained for >=%d weeks: proceed with search'
                                  %require_sust_weeks)


                    if len(conf_t)>0:
                        #### #_conf_#
                        # For each m in conf_weeks, only keep the earliest available confirmation visit
                        # if event_type[-1] != 'PIRA':
                        conf_idx = [np.min([x for x in conf_t[m] if x in conf_idx]) for m in conf_t.keys()]
                        # else:
                        #     conf_idx = [max(-1 if m not in pconf_t.keys() else np.min([x for x in conf_t[m] if x in pconf_idx]),
                        #                 np.min([x for x in conf_t[m] if x in conf_idx])) for m in conf_t.keys()]
                        # conf_idx = [x for x in conf_idx if x>-1]
                        #### #_conf_#

                        next_change = next((x for x in range(conf_idx[0]+1,nvisits) #_r_# #conf_idx[-1]
                            if not isevent_loc(data_id.loc[x,value_col], bl[value_col], type='prog',
                                               baseline_delta=data_id.loc[bl_idx,valuedelta_col]) #_c_# # either increase not sustained
                            or isevent_loc(data_id.loc[x,value_col], data_id.loc[conf_idx[0],value_col], #_r_# #conf_idx[-1]
                                           type='change', baseline_delta=data_id.loc[bl_idx,valuedelta_col]) #_c_# # or further valid change from confirmation
                                        ), None)
                        next_change_ev = next((x for x in range(change_idx+1,nvisits) #_r_#
                            if isevent_loc(data_id.loc[x,value_col], bl[value_col], type='change',
                                           baseline_delta=data_id.loc[bl_idx,valuedelta_col]) # change from event
                                        ), None)

                    if len(conf_t)==0 or not include_event:  # #_rrebl_# or (phase==1 and len(event_type)==nev):
                        search_idx = change_idx + 1
                    elif baseline in ('roving', 'roving_wors'): # #_rrebl_# and phase==0
                        bl_idx = conf_idx[0] # set new baseline at first confirmation time
                        search_idx = nvisits if next_change is None else next_change
                    elif (event_type[-1]!='PIRA' and event=='firstPIRA') or (event_type[-1]!='RAW' and event=='firstRAW'):  # #_rrebl_# and phase==0
                        search_idx = nvisits if next_change_ev is None else next_change_ev #_r_#
                    else:
                        search_idx = nvisits if next_change is None else next_change #next_nonsust
                    if verbose == 2: # and phase == 0: #_rrebl_#
                        print('Baseline at visit no.%d, searching for events from visit no.%s on'
                              %(bl_idx+1, '-' if search_idx>=nvisits else search_idx+1))


                # Confirmed sub-threshold progression: RE-BASELINE
                # ------------------------------------------------
                elif (len(conf_idx) > 0 # confirmation visits available
                        and data_id.loc[change_idx,value_col] > bl[value_col] # value increased from baseline
                        and (all([data_id.loc[x,value_col] > bl[value_col]
                                 for x in range(change_idx+1,conf_idx[0]+1)]) # increase is confirmed
                        if check_intermediate else data_id.loc[conf_idx[0],value_col] > bl[value_col])
                        and baseline in ('roving', 'roving_wors') and sub_threshold
                        ########( #_rrebl_#
                        # and phase == 0 # skip if re-checking for PIRA after post-relapse re-baseline
                        ########)
                        ):
                    next_change = next((x for x in range(conf_idx[0]+1,nvisits)
                        if data_id.loc[x,value_col] <= bl[value_col]), None)
                    bl_idx = conf_idx[0] # set new baseline at first confirmation time
                    search_idx = nvisits if next_change is None else next_change
                    if verbose == 2:
                        print('Confirmed sub-threshold %s progression (visit no.%d)'
                              %(outcome.upper(), change_idx+1))
                        print('New settings: baseline at visit no.%d, searching for events from visit no.%d on'
                              %(bl_idx+1, search_idx+1))

                # NO confirmation:
                # ----------------
                else:
                    search_idx = change_idx + 1 # skip the change and look for other patterns after it
                    if verbose == 2:
                        print('Change not confirmed: proceed with search')


            if (relapse_rebl and proceed  # and phase==0 and len(relapse_dates)>0 and not proceed and nrel>0  #_rrebl_#
                and search_idx < nvisits and ((data_id.loc[bl_idx, date_col] < relapse_dates)
                     & (relapse_dates <= data_id.loc[search_idx, date_col]
                        + (0 if event=='firstPIRA' else relapse_assoc))).any()
                ):
                proceed = 1
                #####( #_rrebl_#
                # phase = 1
                bl_idx = next((x for x in range(bl_idx, nvisits) # visits from current baseline
                               if relapse_dates[irel] <= data_id.loc[x, date_col]  # after `irel`-th relapse
                               ),
                              None)
                if bl_idx is not None:
                    search_idx = bl_idx + 1
                    if verbose == 2:
                        print('Post-relapse rebaseline: baseline at visit no.%d, searching for events from visit no.%d on'
                              %(bl_idx+1, search_idx+1))
                if proceed and (bl_idx is None or bl_idx > nvisits - 2):
                    proceed = 0
                    if verbose == 2:
                        print('Not enough visits after current baseline: end process')
                irel += 1
                # bl_idx = 0
                # search_idx = 1
                #####)

            if proceed and (
                (event == 'first' and len(event_type)>1)
                or (event == 'firsteach' and ('impr' in event_type) and ('prog' in event_type))
                or (event == 'firstprog' and (('RAW' in event_type) or ('PIRA' in event_type) or ('prog' in event_type)))
                or (event == 'firstprogtype' and ('RAW' in event_type) and ('PIRA' in event_type) and ('prog' in event_type))
                or (event == 'firstPIRA' and ('PIRA' in event_type))
                or (event == 'firstRAW' and ('RAW' in event_type))
                        ):
                    proceed = 0
                    if verbose == 2:
                        print('\'%s\' events already found: end process' %event)

            #############(#_rrebl_#
            # if (proceed and search_idx <= nvisits-1 and relapse_rebl and phase == 1
            #         and ((data_id.loc[bl_idx,date_col]<=relapse_dates)
            #              & (relapse_dates<=data_id.loc[search_idx,date_col])).any()): # if search_idx has been moved after another relapse
            #
            #     bl_last = bl[value_col]
            #     bl_idx = next((x for x in range(bl_idx+1,nvisits) # visits after current baseline (or after last confirmed PIRA)
            #                 if ((data_id.loc[bl_idx, date_col]<=relapse_dates)
            #                         & (relapse_dates<=data_id.loc[x, date_col])).any() # after a relapse
            #                 and data_id.loc[x,'closest_rel-'] >= relapse_to_bl # occurring at least `relapse_to_bl` days from last relapse
            #                    ),
            #                 None)
            #
            #     if bl_idx is not None:
            #         ########## Kappos2020 (by Sean Yu)
            #         if bl_last > data_id.loc[bl_idx,value_col]:
            #             data_id.loc[bl_idx, value_col] = bl_last
            #         #########
            #         search_idx = bl_idx + 1
            #         if verbose == 2:
            #             print('Post-relapse rebaseline: baseline at visit no.%d, searching for events from visit no.%d on'
            #                   %(bl_idx+1, search_idx+1))
            #
            #     if proceed and (bl_idx is None or bl_idx > nvisits-2):
            #         proceed = 0
            #         if verbose == 2:
            #             print('Not enough visits after current baseline: end process')
            #
            # elif (proceed and search_idx <= nvisits-1 and relapse_rebl and phase == 1
            #     and not ((data_id.loc[bl_idx,date_col]<=relapse_dates)
            #              & (relapse_dates<=data_id.loc[search_idx,date_col])).any()
            #     and verbose == 2):
            #     print('Post-relapse re-baseline: baseline at visit no.%d, searching for events from visit no.%d on'
            #                   %(bl_idx+1, search_idx+1))
        #############)

        subj_index = results[results[subj_col]==subjid].index

        if len(event_type)>1:

            event_type = event_type[1:] # remove first empty event


            # Spot duplicate events
            # (can only occur if relapse_rebl is enabled - in that case, only keep last detected)
            event_index = np.array(event_index)
            uevents, ucounts = np.unique(event_index, return_counts=True)
            duplicates = [uevents[i] for i in range(len(uevents)) if ucounts[i]>1]
            diff = len(event_index) - len(np.unique(event_index)) # keep track of no. duplicates
            for ev in duplicates:
                all_ind = np.where(event_index==ev)[0]
                event_index[all_ind[:-1]] = -1 # mark duplicate events (all except last) with -1

            event_order = np.argsort(event_index)
            event_order = event_order[diff:] # eliminate duplicates (those marked with -1)

            event_type = [event_type[i] for i in event_order]

            if event.startswith('first'):
                impr_idx = next((x for x in range(len(event_type)) if event_type[x]=='impr'), None)
                prog_idx = next((x for x in range(len(event_type)) if event_type[x] in ('prog', 'RAW', 'PIRA')), None)
                raw_idx = next((x for x in range(len(event_type)) if event_type[x]=='RAW'), None)
                pira_idx = next((x for x in range(len(event_type)) if event_type[x]=='PIRA'), None)
                undef_prog_idx = next((x for x in range(len(event_type)) if event_type[x]=='prog'), None)
                if event=='firsteach':
                    first_events = [impr_idx, prog_idx]
                elif event=='firstprog':
                    first_events = [prog_idx]
                elif event=='firstprogtype':
                    first_events = [raw_idx, pira_idx, undef_prog_idx]
                elif event=='firstPIRA':
                    first_events = [pira_idx]
                elif event=='firstRAW':
                    first_events = [raw_idx]
                first_events = [0] if event=='first' else np.unique([
                    ii for ii in first_events if ii is not None]) # np.unique() returns the values already sorted
                event_type = [event_type[ii] for ii in first_events]
                event_order = [event_order[ii] for ii in first_events]

            if include_stable and len(event_type)==0:
                results.drop(subj_index[1:], inplace=True)
                results.loc[results[subj_col]==subjid, 'nevent'] = 0
                results.loc[results[subj_col]==subjid, 'total_fu'] = total_fu[subjid]
                results.loc[results[subj_col]==subjid, 'time2event'] = total_fu[subjid]
                results.loc[results[subj_col]==subjid, 'date'] = global_start + datetime.timedelta(
                                                days=data_id.loc[nvisits-1,date_col].item())
                results.loc[results[subj_col]==subjid, 'event_type'] = ''
            elif len(event_type)==0:
                results.drop(subj_index, inplace=True)
            else:
                results.drop(subj_index[len(event_type):], inplace=True)
                results.loc[results[subj_col]==subjid, 'event_type'] = event_type
                results.loc[results[subj_col]==subjid, 'bldate'] = [bldate[i] for i in event_order]
                results.loc[results[subj_col]==subjid, 'blvalue'] = [blvalue[i] for i in event_order]
                results.loc[results[subj_col]==subjid, 'date'] = [edate[i] for i in event_order]
                results.loc[results[subj_col]==subjid, 'value'] = [evalue[i] for i in event_order]
                results.loc[results[subj_col]==subjid, 'total_fu'] = total_fu[subjid]
                results.loc[results[subj_col]==subjid, 'time2event'] = [time2event[i] for i in event_order]
                results.loc[results[subj_col]==subjid, 'bl2event'] = [bl2event[i] for i in event_order]
                for m in conf_weeks:
                    results.loc[results[subj_col]==subjid, 'conf'+str(m)] = [conf[m][i] for i in event_order]
                results.loc[results[subj_col]==subjid, 'sust_days'] = [sustd[i] for i in event_order]
                results.loc[results[subj_col]==subjid, 'sust_last'] = [sustl[i] for i in event_order]
                for m in conf_weeks: #[1:]: #_piraconf_#
                    results.loc[results[subj_col]==subjid, 'PIRA_conf'+str(m)] = [pira_conf[m][i] for i in event_order]

        elif include_stable:
            results.drop(subj_index[1:], inplace=True)
            results.loc[results[subj_col]==subjid, 'nevent'] = 0
            results.loc[results[subj_col]==subjid, 'total_fu'] = total_fu[subjid]
            results.loc[results[subj_col]==subjid, 'time2event'] = total_fu[subjid]
            results.loc[results[subj_col]==subjid, 'date'] = global_start + datetime.timedelta(
                                            days=data_id.loc[nvisits-1,date_col].item())

        else:
            results.drop(subj_index, inplace=True)

        improvement = (results.loc[results[subj_col]==subjid, 'event_type']=='impr').sum()
        progression = results.loc[results[subj_col]==subjid, 'event_type'].isin(('prog', 'RAW', 'PIRA')).sum()
        undefined_prog = (results.loc[results[subj_col]==subjid, 'event_type']=='prog').sum()
        RAW = (results.loc[results[subj_col]==subjid, 'event_type']=='RAW').sum()
        PIRA = (results.loc[results[subj_col]==subjid, 'event_type']=='PIRA').sum()
        summary.loc[subjid, ['event_sequence', 'improvement', 'progression',
                'RAW', 'PIRA', 'undefined_prog']] = [', '.join(event_type), improvement, progression,
                                                     RAW, PIRA, undefined_prog]
        if event.startswith('firstprog'):
            summary.drop(columns=['improvement'], inplace=True)

        if verbose == 2:
            print('Event sequence: %s' %(', '.join(event_type) if len(event_type)>0 else '-'))

    if verbose>=1:
        print('\n---\nOutcome: %s\nConfirmation %s: %s weeks (-%d days, +%s)\nBaseline: %s%s%s\n'\
              'Relapse influence (baseline): %d days\nRelapse influence (event): %d days\n'\
              'Relapse influence (confirmation): %d days\nEvents detected: %s'
          %(outcome.upper(), 'over' if check_intermediate else 'at',
            conf_weeks, conf_tol_days[0], 'inf' if conf_unbounded_right else str(conf_tol_days[1])+' days',
            baseline, ' (sub-threshold)' if sub_threshold else '',
            ' (and post-relapse re-baseline)' if relapse_rebl else '',
            relapse_to_bl, relapse_to_event, relapse_to_conf, event))
        print('---\nTotal subjects: %d\n---\nProgressed: %d (PIRA: %d; RAW: %d)'
              %(nsub, (summary['progression']>0).sum(),
                (summary['PIRA']>0).sum(), (summary['RAW']>0).sum()))
        if event not in ('firstprog', 'firstprogtype', 'firstPIRA', 'firstRAW'):
            print('Improved: %d' %(summary['improvement']>0).sum())
        if event in ('multiple', 'firstprogtype'):
            print('---\nProgression events: %d (PIRA: %d; RAW: %d)'
                  %(summary['progression'].sum(),
                    summary['PIRA'].sum(), summary['RAW'].sum()))
        if event=='multiple':
            print('Improvement events: %d' %(summary['improvement'].sum()))

        if min_value > -np.inf:
            print('---\n*** WARNING only progressions to %s>=%d are considered ***'
                  %(outcome.upper(), min_value))

    columns = results.columns
    if not include_dates:
        columns = [c for c in columns if not c.endswith('date')]
    if not include_value:
        columns = [c for c in columns if not c.endswith('value')]
    results = results[columns]

    for w in warning_msgs:
        warnings.warn(w)

    return summary, results.reset_index(drop=True)


#####################################################################################

def col_to_date(column, format=None, remove_na=False):
    """
    Convert dataframe column into datetime.date format.
    Arguments:
     column: the dataframe column to convert
     format: date format (see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)
    Returns:
     the column converted to dataframe.date
    """

    if format is None:
        infer_datetime_format = True

    vtype = np.vectorize(lambda x: type(x))

    if remove_na:
        column.dropna(inplace=True)

    column_all = column.copy().astype('datetime64[D]')
    naidx = pd.Series([False]*len(column), index=column.index) if remove_na else column.isna()
    column = column.dropna()

    if all([d is pd.Timestamp for d in vtype(column)]):
        dates = column.dt.date
    elif all([d is datetime.datetime for d in vtype(column)]):
        dates = column.apply(lambda x : x.date())
    elif all([d is np.datetime64 for d in vtype(column)]):
        dates = column.astype('datetime64[D]').apply(lambda x : x.date())
    elif not all([d is datetime.date for d in vtype(column)]):
        dates = pd.to_datetime(column, format=format, infer_datetime_format=infer_datetime_format).dt.date
    else:
        dates = column

    column_all.loc[~naidx] = dates

    return column_all.dt.date


#####################################################################################

def age_column(date, dob, col_name='Age', remove_na=False):
    """
    Compute difference between two columns of dates.
    Arguments:
     date: end date.
     dob: start date.
     col_name: name of new column.
     remove_na: whether to remove NaN entries.
    Returns:
     difference in days.
    """
    #date = col_to_date(date)
    #dob = col_to_date(dob)
    date = pd.to_datetime(date)
    dob = pd.to_datetime(dob)


    if not remove_na:
        diff = pd.Series(np.nan, index=date.index, name=col_name)

    naidx = (date.isna()) | (dob.isna())
    date, dob = date.loc[~naidx].copy(), dob.loc[~naidx].copy()
    if remove_na:
        diff = pd.Series(np.nan, index=date.index, name=col_name)

    dob.loc[(dob.apply(lambda x: x.day)==29)
            & (dob.apply(lambda x: x.month)==2)] = dob.loc[(dob.apply(lambda x: x.day)==29)
            & (dob.apply(lambda x: x.month)==2)].apply(lambda dt: dt.replace(day=28))

    this_year_birthday = pd.to_datetime(dict(year=date.apply(lambda x: x.year),
                                             day=dob.apply(lambda x: x.day),
                                             month=dob.apply(lambda x: x.month)))
    diff_tmp = date.apply(lambda x: x.year) - dob.apply(lambda x: x.year)
    diff_tmp.loc[this_year_birthday >= date] = diff_tmp.loc[this_year_birthday >= date] - 1

    diff.loc[~naidx] = diff_tmp

    return diff


#####################################################################################

def compute_delta(baseline, outcome='edss'):
    """
    Definition of progression deltas for different tests.
    Arguments:
     baseline: baseline value
     outcome: type of test ('edss'[default],'nhpt','t25fw','sdmt')
    Returns:
     minimum delta corresponding to valid change
    """
    if outcome == 'edss':
        if baseline>=0 and baseline<.5:
            return 1.5
        elif baseline>=.5 and baseline<5.5:
            return 1.0
        elif baseline>=5.5 and baseline<=10:
            return 0.5
        else:
            raise ValueError('invalid EDSS score')
    elif outcome in ('nhpt', 't25fw'):
        if baseline<0:
            raise ValueError('invalid %s score' %outcome.upper())
        if outcome=='nhpt' and baseline>300:
            warnings.warn('NHPT score >300')
        if outcome=='t25fw' and baseline>180:
            warnings.warn('T25FW score >180')
        return baseline/5
    elif outcome == 'sdmt':
        if baseline<0 or baseline>110:
            raise ValueError('invalid SDMT score')
        return min(baseline/10, 3)
    else:
        raise Exception('outcome must be one of: \'edss\',\'nhpt\',\'t25fw\',\'sdmt\'')


#####################################################################################

def is_event(x, baseline, type, outcome=None, worsening=None,
             sub_threshold=False, delta_fun=None, baseline_delta=None):
    """
    Check for change from baseline.
    Arguments:
     x: new value
     baseline: baseline value
     type: 'prog' or 'impr' or 'change'
     outcome: type of test (one of: 'edss','nhpt','t25fw','sdmt')
     worsening: 'increase' or 'decrease'. If outcome is specified, it is automatically assigned
                ('increase' for edss, nhpt, t25fw; 'decrease' for sdmt)
     baseline_delta: baseline value to use for delta, if different from baseline
    Returns:
     True if event else False
    """
    if baseline_delta is None:
        baseline_delta = baseline
    if outcome in ('edss','nhpt','t25fw'):
        worsening = 'increase'
    elif outcome=='sdmt':
        worsening = 'decrease'
    elif worsening is None:
        raise ValueError('Either specify a valid outcome type, or specify worsening direction')
    improvement = 'increase' if worsening=='decrease' else 'decrease'
    if sub_threshold:
        event_sign = {'increase' : x > baseline, 'decrease' : x < baseline, 'change' : x != baseline}
    else:
        if delta_fun is None:
            fun_tmp = compute_delta
        else:
            def fun_tmp(baseline, outcome):
                try:
                    return delta_fun(baseline, outcome)
                except TypeError:
                    return delta_fun(baseline)
        event_sign = {'increase' : x - baseline >= fun_tmp(baseline_delta, outcome),
                 'decrease' : x - baseline <= - fun_tmp(baseline_delta, outcome),
                 'change' : abs(x - baseline) >= fun_tmp(baseline_delta, outcome)}
    event = {'prog' : event_sign[worsening], 'impr': event_sign[improvement], 'change' : event_sign['change']}
    return event[type]


#####################################################################################

def value_milestone(data, milestone, value_col, date_col, subj_col,
                   relapse=None, rsubj_col=None, rdate_col=None,
                   conf_weeks=24, conf_tol_days=45, conf_unbounded_right=False,
                   relapse_to_event=0, relapse_to_conf=30,
                   verbose=0):
    """
    ARGUMENTS:
        data, DataFrame: longitudinal data containing subject ID, outcome value, date of visit
        milestone, float: value to check
        subj_col, str: name of data column with subject ID
        value_col, str: name of data column with outcome value
        date_col, str: name of data column with date of visit
        relapse, DataFrame: (optional) longitudinal data containing subject ID and relapse date
        rsubj_col / rdate_col, str: name of columns for relapse data, if different from outcome data
        conf_weeks, int or list-like : period before confirmation (weeks)
        conf_tol_days, int or list-like of length 1 or 2: tolerance window for confirmation visit (days): [t(months)-conf_tol[0](days), t(months)+conf_tol[0](days)]
        conf_unbounded_right, bool: if True, confirmation window is [t(months)-conf_tol(days), inf)
        relapse_to_event, int: minimum distance from a relapse (days) for an outcome value to be considered valid.
        relapse_to_conf, int: minimum distance from a relapse (days) for a visit to be a valid confirmation visit.
        verbose, int: 0[default, print no info], 1[print concise info], 2[print extended info]
    RETURNS:
        DataFrame containing:
         - date of first reaching value >=milestone (or last date of follow-up if milestone is not reached);
         - first value >=milestone, if present, otherwise last value recorded
    """

    try:
       _ = (e for e in conf_weeks) # check if conf_weeks is iterable
    except TypeError:
       conf_weeks = [conf_weeks] # if it's not, make it a list with a single element


    if relapse is not None and rsubj_col is None:
        rsubj_col = subj_col
    if relapse is not None and rdate_col is None:
        rdate_col = date_col

    # Remove missing values from columns of interest
    data = data[[subj_col, value_col, date_col]].dropna()

    # Convert dates to datetime.date format
    data[date_col] = pd.to_datetime(data[date_col]) #col_to_date(data[date_col]) #
    if relapse is None:
        relapse = pd.DataFrame([], columns=[rsubj_col, rdate_col])
        relapse_start = data[date_col].min()
    else:
        relapse = relapse[[rsubj_col, rdate_col]].copy().dropna() # remove missing values from columns of interest
        relapse[rdate_col] = pd.to_datetime(relapse[rdate_col]) #col_to_date(relapse[rdate_col]) #
        relapse_start = relapse[rdate_col].min()
    # Convert dates to days from minimum #_d_#
    global_start = min(data[date_col].min(), relapse_start)
    relapse[rdate_col] = (relapse[rdate_col] - global_start).apply(lambda x : x.days)
    data[date_col] = (data[date_col] - global_start).apply(lambda x : x.days)


    # conf_window = (int(conf_weeks*7) - conf_tol_days, float('inf') if conf_unbounded_right
    #                else int(conf_weeks*7) + conf_tol_days)
    conf_window = [(int(c*7) - conf_tol_days[0], float('inf')) if conf_unbounded_right
                       else (int(c*7) - conf_tol_days[0], int(c*7) + conf_tol_days[1]) for c in conf_weeks]

    all_subj = data[subj_col].unique()
    nsub = len(all_subj)
    results = pd.DataFrame([[None, None]]*nsub, columns=[date_col, value_col], index=all_subj)

    for subjid in all_subj:
        data_id = data.loc[data[subj_col]==subjid,:].copy()

        udates, ucounts = np.unique(data_id[date_col].values, return_counts=True)
        if any(ucounts>1):
            data_id = data_id.groupby(date_col).last().reset_index()
            # groupby() indexes the dataframe by date_col: resetting index to convert date_col back into a normal column


        data_id.reset_index(inplace=True, drop=True)

        nvisits = len(data_id)
        if verbose > 0:
            print('\nSubject #%s: %d visit%s'
              %(subjid,nvisits,'' if nvisits==1 else 's'))
            if any(ucounts>1):
                print('Found multiple visits in the same day: only keeping last')
        first_visit = data_id[date_col].min()
        if relapse is not None:
            relapse_id = relapse.loc[relapse[rsubj_col]==subjid,:].reset_index(drop=True)
            relapse_id = relapse_id.loc[relapse_id[rdate_col] >= first_visit - relapse_to_event,:] #_d_#datetime.timedelta(days=relapse_to_event)
                                                                # ignore relapses occurring before first visit
            relapse_dates = relapse_id[rdate_col].values
            relapse_df = pd.DataFrame([relapse_dates]*len(data_id))
            relapse_df['visit'] = data_id[date_col].values
            dist = relapse_df.drop(['visit'],axis=1).subtract(relapse_df['visit'], axis=0) #_d_# .apply(lambda x : pd.to_timedelta(x).dt.days)
            distm = - dist.mask(dist>0, other= - float('inf'))
            distp = dist.mask(dist<0, other=float('inf'))
            data_id['closest_rel-'] = float('inf') if all(distm.isna()) else distm.min(axis=1)
            data_id['closest_rel+'] = float('inf') if all(distp.isna()) else distp.min(axis=1)
        else:
            data_id['closest_rel-'] = float('inf')
            data_id['closest_rel+'] = float('inf')

        proceed = 1
        search_idx = 0
        while proceed:
            milestone_idx = next((x for x in range(search_idx,nvisits)
                    if data_id.loc[x,value_col]>=milestone # first occurring value>=milestone
                    and (data_id.loc[x, 'closest_rel-'] >= relapse_to_event)), None) # occurring at least `relapse_to_event` days from last relapse
            if milestone_idx is None: # value does not change in any subsequent visit
                results.at[subjid,date_col] = global_start + datetime.timedelta(days=data_id.iloc[-1,:][date_col].item()) #_d_# data_id.iloc[-1,:][date_col]
                results.at[subjid,value_col] = data_id.iloc[-1,:][value_col]
                proceed = 0
                if verbose == 2:
                    print('No value >=%d in any visit: end process' %(milestone))
            else:
                conf_idx = [[x for x in range(milestone_idx+1, nvisits)
                        if c[0] <= data_id.loc[x,date_col] - data_id.loc[milestone_idx,date_col] <= c[1] # date in confirmation range
                        and data_id.loc[x,'closest_rel-'] >= relapse_to_conf] # occurring at least `relapse_to_conf` days from last relapse
                        for c in conf_window]
                conf_idx = np.unique([x for i in range(len(conf_idx)) for x in conf_idx[i]])
                if len(conf_idx)>0 and all([data_id.loc[x,value_col]
                            >= milestone for x in range(milestone_idx+1,conf_idx[0]+1)]):
                    results.at[subjid,date_col] = global_start + datetime.timedelta(days=data_id.loc[milestone_idx,date_col].item()) #_d_# #data_id.loc[milestone_idx,date_col]
                    results.at[subjid,value_col] = data_id.loc[milestone_idx,value_col]
                    proceed = 0
                    if verbose == 2:
                        print('Found value >=%d: end process' %(milestone))
                else:
                    next_change = next((x for x in range(milestone_idx+1,nvisits)
                    if data_id.loc[x,value_col]<milestone), None)
                    search_idx = search_idx + 1 if next_change is None else next_change + 1
                    if verbose == 2:
                        print('value >=%d not confirmed: continue search' %(milestone))

        if results.at[subjid,date_col] is None:
            results.at[subjid,date_col] = global_start + datetime.timedelta(days=data_id.iloc[-1,:][date_col] .item()) #_d_# data_id.iloc[-1,:][date_col]
            results.at[subjid,value_col] = data_id.iloc[-1,:][value_col]

    return results


#####################################################################################


def separate_ri_ra(data, relapse, mode, value_col, date_col, subj_col,
                   rsubj_col=None, rdate_col=None, outcome='edss', delta_fun=None,
                   conf_weeks=24, conf_tol_days=45, conf_unbounded_right=False, require_sust_weeks=0,
                   relapse_to_bl=30, relapse_to_event=0, relapse_to_conf=30, relapse_assoc=90,
                   subtract_bl=False, drop_orig=False, return_rel_num=False, return_raw_dates=False, verbose=0):
    """
    ARGUMENTS:
        data, DataFrame: longitudinal data containing subject ID, outcome value, date of visit
        relapse, DataFrame: longitudinal data containing subject ID and relapse date
        mode, str: 'ri' (relapse-independent), 'ra' (relapse-associated), 'both', 'none'
        subj_col, str: name of data column with subject ID
        value_col, str: name of data column with outcome value
        date_col, str: name of data column with date of visit
        rsubj_col / rdate_col, str: name of columns for relapse data, if different from outcome data
        relapse_to_bl, int: minimum distance from a relapse (days) for a visit to be used as baseline (otherwise the next available visit is used as baseline).
        relapse_to_event, int: minimum distance from a relapse (days) for an event to be considered as such.
        relapse_to_conf, int: minimum distance from a relapse (days) for a visit to be a valid confirmation visit.
        relapse_assoc, int: maximum distance from last relapse for a progression event to be considered as RAW (days)
        outcome, str: 'edss'[default],'nhpt','t25fw','sdmt'
        delta_fun, function: (optional) Custom function specifying the minimum delta corresponding to a valid change from baseline.
        conf_weeks, int or list-like: period before confirmation (weeks)
        conf_tol_days, int or list-like of length 1 or 2: tolerance window for confirmation visit (days): [t(months)-conf_tol[0](days), t(months)+conf_tol[0](days)]
        conf_unbounded_right, bool: if True, confirmation window is [t(months)-conf_tol(days), inf)
        require_sust_weeks, int: count an event as such only if sustained for _ weeks from confirmation
        subtract_bl, bool: if True, report values as deltas relative to baseline value
        drop_orig, bool: if True, replace original value column with relapse-independent / relapse-associated version
        return_rel_num, bool:
        return_raw_dates, bool:
        verbose, int: 0[default, print no info], 1[print concise info], 2[print extended info]
    RETURNS:
        the original DataFrame, plus the additional columns - i.e. the ones enabled among:
         - cumulative relapse-independent values
         - cumulative relapse-associated values
         - cumulative relapse number
         - RAW event dates.
    """

    try:
       _ = (e for e in conf_weeks) # check if conf_weeks is iterable
    except TypeError:
       conf_weeks = [conf_weeks] # if it's not, make it a list with a single element


    if rsubj_col is None:
        rsubj_col = subj_col
    if rdate_col is None:
        rdate_col = date_col
    if isinstance(conf_tol_days, int):
        conf_tol_days = [conf_tol_days, conf_tol_days]
    if isinstance(relapse_to_bl, int):
        relapse_to_bl = [relapse_to_bl, 0]

    data_sep = data.copy()
    relapse = relapse.copy()

    # Remove missing values from columns of interest
    data_sep = data_sep.loc[~data_sep[[subj_col, value_col, date_col]].isna().any(axis=1), :].dropna()
    # Convert dates to datetime.date format
    data_sep[date_col] = pd.to_datetime(data_sep[date_col]) #col_to_date(data_sep[date_col]) #
    if relapse is None:
        relapse = pd.DataFrame([], columns=[rsubj_col, rdate_col])
        relapse_start = data_sep[date_col].min()
    else:
        relapse = relapse[[rsubj_col, rdate_col]].copy().dropna() # remove missing values from columns of interest
        relapse[rdate_col] = pd.to_datetime(relapse[rdate_col]) #col_to_date(relapse[rdate_col]) #
        relapse_start = relapse[rdate_col].min()
    # Convert dates to days from minimum #_d_#
    global_start = min(data[date_col].min(), relapse_start)
    relapse[rdate_col] = (relapse[rdate_col] - global_start).apply(lambda x : x.days)
    data_sep[date_col] = (data_sep[date_col] - global_start).apply(lambda x : x.days)

    ri_col, ra_col, bump_col = 'ri'+value_col, 'ra'+value_col, value_col+'_bumps'
    if mode!='none':
        data_sep[ra_col] = 0
        data_sep[bump_col] = 0
    if mode in ('ri', 'both'):
        data_sep[ri_col] = data_sep[value_col]
    if return_rel_num:
        data_sep['relapse_num'] = 0
    if subtract_bl:
        data_sep[value_col+'-bl'] = 0
        if mode in ('ri', 'both'):
            data_sep[ri_col+'-bl'] = 0

    def delta(value):
        if delta_fun is None:
            return .5 if outcome=='edss' else 1 #compute_delta(value, outcome) #
        else:
            return delta_fun(value)

    # conf_window = (int(conf_weeks*7) - conf_tol_days[0],
    #                float('inf')) if conf_unbounded_right else (int(conf_weeks*7) - conf_tol_days[0],
    #                                                 int(conf_weeks*7) + conf_tol_days[1])
    conf_window = [(int(c*7) - conf_tol_days[0], float('inf')) if conf_unbounded_right
                       else (int(c*7) - conf_tol_days[0], int(c*7) + conf_tol_days[1]) for c in conf_weeks]

    all_subj = data[subj_col].unique()
    nsub = len(all_subj)

    if return_raw_dates:
        raw_events = []

    for subjid in all_subj:
        data_id = data_sep.loc[data_sep[subj_col]==subjid,:].copy().reset_index(drop=True)
        nvisits = len(data_id)

        relapse_id = relapse.loc[relapse[rsubj_col]==subjid,:].reset_index(drop=True)

        relapse_dates = relapse_id[rdate_col].values
        relapse_df = pd.DataFrame([relapse_dates]*len(data_id))
        relapse_df['visit'] = data_id[date_col].values
        dist = relapse_df.drop(['visit'],axis=1).subtract(relapse_df['visit'], axis=0) #_d_# .apply(lambda x : pd.to_timedelta(x).dt.days)
        distm = - dist.mask(dist>0, other= - float('inf'))
        distp = dist.mask(dist<0, other=float('inf'))
        data_id['closest_rel-'] = float('inf') if all(distm.isna()) else distm.min(axis=1)
        data_id['closest_rel+'] = float('inf') if all(distp.isna()) else distp.min(axis=1)

        # First visit out of relapse influence
        rel_free_bl = next((x for x in range(len(data_id))
                        if data_id.loc[x,'closest_rel-'] > relapse_to_bl[0]
                        and data_id.loc[x,'closest_rel+'] > relapse_to_bl[1]), None)

        if len(data_id)>0:
            nrel = len(relapse_id) if len(data_id)==0 else sum(relapse_id[rdate_col] >= data_id.loc[0, date_col])
        if verbose > 0:
            print('\nSubject #%s: %d visit%s, %d relapse%s'
              %(subjid, nvisits, '' if nvisits==1 else 's', nrel, '' if nrel==1 else 's'))


        if mode != 'none':

            if rel_free_bl is None:
                # glob_bl_idx = data_id.sort_values(by=value_col).index[0] # minimum
                # global_bl = data_id.loc[glob_bl_idx, :].copy()
                data_id = data_id.loc[[],:].reset_index(drop=True)
                # bump = data_id[value_col] - data_id[value_col].min() # values exceeding the minimum
                # data_id[bump_col] = data_id[bump_col] + bump
                if verbose==2:
                    print('No baseline visits out of relapse influence')
            elif rel_free_bl > 0:
                glob_bl_idx = data_id.loc[:rel_free_bl,:].sort_values(by=value_col).index[0]
                global_bl = data_id.loc[glob_bl_idx, :].copy()
                data_id = data_id.loc[rel_free_bl:,:].reset_index(drop=True)
                # bump = data_id[value_col] - data_id.loc[:rel_free_bl, value_col].min() # values exceeding the minimum up to the baseline
                # data_id.loc[:rel_free_bl-1, bump_col] = data_id.loc[:rel_free_bl-1, bump_col] + bump.loc[:rel_free_bl-1]
                if verbose==2:
                    print('Moving baseline to first visit out of relapse influence (%dth visit)' %(rel_free_bl+1))
            else:
                glob_bl_idx = 0
                global_bl = data_id.loc[0, :].copy()
            nvisits = len(data_id)

            bl_date = data_id[date_col].min() #data_id[date_col].max() if rel_free_bl is None else data_id.loc[rel_free_bl, date_col] #
            relapse_id = relapse_id.loc[relapse_id[rdate_col] > bl_date, #_d_# datetime.timedelta(days=relapse_to_bl)
                            :].reset_index(drop=True) # ignore relapses occurring before or at baseline
            if rel_free_bl is not None and rel_free_bl > 0 and verbose==2:
                print('Relapses left to analyse: %d' %len(relapse_id))

            ##########
            visit_dates = data_id[date_col].values
            relapse_df = pd.DataFrame([visit_dates]*len(relapse_id))
            relapse_df['relapse'] = relapse_id[rdate_col].values
            dist = relapse_df.drop(['relapse'],axis=1).subtract(relapse_df['relapse'], axis=0) #_d_# .apply(lambda x : pd.to_timedelta(x).dt.days)
            distm = - dist.mask(dist>0, other=float('nan'))
            distp = dist.mask(dist<0, other=float('nan'))
            relapse_id['closest_vis-'] = None if all(distm.isna()) else distm.idxmin(axis=1)
            relapse_id['closest_vis+'] = None if all(distp.isna()) else distp.idxmin(axis=1)
            ##########

            # if nvisits>0:
            #     global_bl = data_id.loc[0,:].copy()
            #     glob_bl_idx = 0

            delta_raw, raw_dates = [], []
            last_conf = None

            for irel in range(len(relapse_id)):

                if last_conf is not None and last_conf>=relapse_id.loc[irel,rdate_col]: #.date()
                    if verbose==2:
                        print('Relapse #%d / %d: skipped (falls within confirmation period of last RAW)'
                              %(irel+1, len(relapse_id)))
                    continue
                if verbose==2:
                    print('Relapse #%d / %d' %(irel+1, len(relapse_id)))
                change_idx = relapse_id.loc[irel,'closest_vis+']

                # Baseline set to last value before the relapse:
                bl_idx = next((int(relapse_id.loc[irel,'closest_vis-'] - n) for n in range(int(relapse_id.loc[irel,'closest_vis-'])) if
                    relapse_id.loc[irel, rdate_col] - data_id.loc[int(relapse_id.loc[irel,'closest_vis-'])-n, date_col] > relapse_to_bl[1]),
                    glob_bl_idx)
                # bl_idx = change_idx-1 if relapse_id.loc[irel,'closest_vis-']==change_idx\
                #     else int(relapse_id.loc[irel,'closest_vis-'])

                bl = data_id.loc[bl_idx, :].copy()
                # If baseline is part of a bump caused by a previous relapse, subtract the bump
                # (unless it ends up below global baseline):
                bl[value_col] = max(bl[value_col] - bl[bump_col], global_bl[value_col])

                # Look at *all* visits within `relapse_assoc` days from relapse and identify first CONFIRMED change (if any)
                confirmed = False
                ch_idx_tmp = change_idx
                while (not confirmed and ~np.isnan(change_idx) and ch_idx_tmp<nvisits
                    and (data_id.loc[ch_idx_tmp,date_col] - relapse_id.loc[irel,rdate_col]) <= relapse_assoc):
                    #change_idx = ch_idx_tmp

                    # Look at *all* visits within `relapse_assoc` days from relapse and identify first change (if any)
                    stable = True
                    #ch_idx_tmp = change_idx
                    while (stable and ~np.isnan(change_idx) and ch_idx_tmp<nvisits
                        and (data_id.loc[ch_idx_tmp,date_col] - relapse_id.loc[irel,rdate_col]) <= relapse_assoc): #_d_# .days # within `relapse_assoc` days from last relapse
                        change_idx = ch_idx_tmp
                        stable = (data_id.loc[change_idx,value_col] - bl[value_col]
                                < delta(bl[value_col])) or ((data_id.loc[ch_idx_tmp,date_col]
                                - relapse_id.loc[irel,rdate_col]) < relapse_to_event) #_d_# .days
                                # no increase, or event within `relapse_to_event` days after last relapse
                        ch_idx_tmp = change_idx + 1

                    if (np.isnan(change_idx) # no change, or
                        or (data_id.loc[change_idx,date_col]
                            - relapse_id.loc[irel,rdate_col]) > relapse_assoc #_d_# .days # change is out of relapse influence, or
                        or data_id.loc[change_idx,value_col] - bl[value_col]
                            < delta(bl[value_col]) # no increase
                        ):
                        if verbose == 2:
                                print('No relapse-associated worsening')
                        confirmed = False
                    else:
                        change_idx = int(change_idx)
                        conf_idx = [[x for x in range(change_idx+1, nvisits)
                            if c[0] <= data_id.loc[x,date_col] - data_id.loc[change_idx,date_col] <= c[1] # date in confirmation range
                            and data_id.loc[x,'closest_rel-'] >= relapse_to_conf] # occurring at least `relapse_to_conf` days from last relapse
                            for c in conf_window]
                        conf_idx = [x for i in range(len(conf_idx)) for x in conf_idx[i]]
                        conf_idx = None if len(conf_idx)==0 else min(conf_idx)

                        confirmed = (conf_idx is not None  # confirmation visits available
                            and all([data_id.loc[x, value_col] - bl[value_col] >= delta(bl[value_col])
                                 for x in range(change_idx + 1, conf_idx + 1)])) # increase is confirmed at first valid date

                        # CONFIRMED PROGRESSION:
                        # ---------------------
                        if confirmed:
                            valid_prog = 1
                            if require_sust_weeks:
                                next_nonsust = next((x for x in range(conf_idx+1,nvisits) # next value found
                                if data_id.loc[x,value_col] - bl[value_col] < delta(bl[value_col]) # increase not sustained
                                                ), None)
                                valid_prog = (next_nonsust is None) or (data_id.loc[next_nonsust,date_col]
                                            - data_id.loc[change_idx,date_col]) > require_sust_weeks*7 #_d_# .days
                            if valid_prog:
                                sust_idx = next((x for x in range(conf_idx+1,nvisits) # next value found
                                            if (data_id.loc[x,date_col] - data_id.loc[change_idx,date_col]) #_d_# .days
                                            > require_sust_weeks*7
                                                ), None)
                                sust_idx = nvisits-1 if sust_idx is None else sust_idx-1 #conf_idx #
                                # Set value change as the minimum within the "sustained" interval before the following relapse:
                                end_idx = max(relapse_id.loc[irel+1,'closest_vis+']-1, conf_idx) if irel<len(relapse_id)-1 else sust_idx
                                # NB: PANDAS SLICING WITH .loc INCLUDES THE RIGHT END!!
                                value_change = data_id.loc[change_idx:end_idx,value_col].min() - bl[value_col]
                                # value_change = data_id.loc[change_idx,value_col] - bl[value_col]
                                # Detect potential bumps:
                                bump = data_id[value_col] - data_id.loc[change_idx:end_idx,value_col].min() # values exceeding the minimum
                                bump = np.maximum(bump.loc[change_idx:conf_idx], 0)
                                data_id.loc[change_idx:conf_idx, bump_col] = data_id.loc[change_idx:conf_idx, bump_col] + bump

                                #_bl_#########
                                # New baseline: minimum between the two relapses
                                # bl_idx = data_id.loc[conf_idx:end_idx].sort_values(by=value_col).index[0]
                                ##############

                                delta_raw.append(value_change)
                                raw_dates.append(data_id.loc[change_idx,date_col])
                                if return_raw_dates:
                                    raw_events.append([subjid, global_start + datetime.timedelta(
                                            days=data_id.loc[change_idx,date_col].item())]) #_d_# data_id.loc[change_idx,date_col]
                                last_conf = data_id.loc[conf_idx,date_col]

                                if verbose == 2:
                                    print('Relapse-associated confirmed progression on %s'
                                          %(global_start + datetime.timedelta(
                                            days=data_id.loc[change_idx,date_col].item())))  #_d_# data_id.loc[change_idx,date_col]
                            else:
                                #end_idx = relapse_id.loc[irel+1,'closest_vis+']-1 if irel<len(relapse_id)-1 else next_nonsust-1
                                # NB: PANDAS SLICING WITH .loc INCLUDES THE RIGHT END!!
                                bump = data_id[value_col] - data_id.loc[next_nonsust,value_col] #data_id.loc[conf_idx+1:end_idx,value_col].max()
                                bump = np.maximum(bump.loc[change_idx:next_nonsust-1], 0) #change_idx:min(conf_idx, end_idx)
                                data_id.loc[change_idx:next_nonsust-1, bump_col] =\
                                    data_id.loc[change_idx:next_nonsust-1, bump_col] + bump
                                if verbose == 2:
                                    print('Change confirmed but not sustained for >=%d weeks: proceed with search'
                                          %require_sust_weeks)

                        # NO confirmation:
                        # ----------------
                        else:
                            end_idx = relapse_id.loc[irel+1,'closest_vis+']-1 if irel<len(relapse_id)-1 else conf_idx
                            if conf_idx is not None and end_idx > change_idx:
                                # NB: PANDAS SLICING WITH .loc INCLUDES THE RIGHT END!!
                                bump = data_id[value_col] - bl[value_col]
                                bump = np.maximum(bump.loc[change_idx:min(conf_idx, end_idx)], 0)
                                data_id.loc[change_idx:min(conf_idx, end_idx), bump_col] =\
                                    data_id.loc[change_idx:min(conf_idx, end_idx), bump_col] + bump
                            if verbose == 2:
                                print('Change not confirmed: proceed with search')

                        # print(bump)
                        # print(data_id[bump_col])

                    ch_idx_tmp = change_idx + 1

            if verbose == 2:
                print('Examined all relapses: end process')

            for d_value, date in zip(delta_raw, raw_dates):
                data_id.loc[data_id[date_col]>=date, ra_col]\
                    = data_id.loc[data_id[date_col]>=date, ra_col] + d_value


        if mode in ('ri', 'both'):
            data_id[ri_col] = np.maximum(data_id[value_col] - data_id[ra_col] - data_id[bump_col], 0)
            if subtract_bl and len(data_id)>0:
                data_id[ri_col+'-bl'] = data_id[ri_col] - global_bl[value_col] #data_id.loc[glob_bl_idx,ri_col]
        if subtract_bl and len(data_id)>0:
                data_id[value_col+'-bl'] = data_id[value_col] - global_bl[value_col] #data_id.loc[glob_bl_idx,value_col]

        if return_rel_num and len(data_id)>0:
            for date in relapse_dates:
                data_id.loc[data_id[date_col] >= date,'relapse_num']\
                    = data_id.loc[data_id[date_col] >= date,'relapse_num'] + 1 #_d_# pd.to_datetime(date).date()

        # Remove rows of dropped visits
        ind = data_sep.index[np.where(data_sep[subj_col]==subjid)[0]]
        ind = ind[:-len(data_id)] if len(data_id)>0 else ind
        data_sep = data_sep.drop(index=ind)

        # Update collective dataframe
        data_sep.loc[data[subj_col]==subjid,:] = data_id.drop(columns=['closest_rel-', 'closest_rel+']).values

    if drop_orig:
        data_sep = data_sep.drop(columns=value_col).rename(columns={ri_col : value_col})
        if mode=='ra':
            data_sep = data_sep.rename(columns={ra_col : value_col})

    data_sep[date_col] = [global_start + datetime.timedelta(
                    days=int(data_sep.loc[ii,date_col])) for ii in data_sep.index]
    data_sep[date_col] = pd.to_datetime(data_sep[date_col])

    return (data_sep, pd.DataFrame(raw_events, columns=[subj_col,date_col])) if return_raw_dates else data_sep


#####################################################################################

def confirmed_value(data, value_col, date_col, idx=0, min_confirmed=None,
                   relapse=None, rdate_col=None,
                   conf_weeks=24, conf_tol_days=45, conf_unbounded_right=False,
                   relapse_to_event=0, relapse_to_conf=30, relapse_indep=None,
                   pira=False):
    """
    ARGUMENTS:
        data, DataFrame: patient follow-up, containing outcome value and date of visit
        value_col, str: name of data column with outcome value
        date_col, str: name of data column with date of visit
        idx, int: index of event to be confirmed in data
        min_confirmed, int: minimum value to be reached in confirmation visits (e.g. baseline+delta)
                            (ignored if >value at event, set to value at event if None is given)
        relapse, DataFrame: optional, relapse dates
        rdate_col, str: name of columns for relapse data, if different from outcome data
        conf_weeks, int or list-like: period before confirmation (weeks)
        conf_tol_days, int or list-like of length 1 or 2: tolerance window for confirmation visit (days): [t(months)-conf_tol[0](days), t(months)+conf_tol[0](days)]
        conf_unbounded_right, bool: if True, confirmation window is [t(months)-conf_tol(days), inf)
        relapse_to_event, int: minimum distance from a relapse (days) for an outcome value to be considered valid.
        relapse_to_conf, int: minimum distance from a relapse (days) for a visit to be a valid confirmation visit.
        relapse_indep, dict: relapse-free intervals around event and confirmation to define PIRA.
                            {'event':(e0,e1), 'conf':(c0,c1)}
                            If the right end is None, the interval is assumed to extend up to the left end of the next interval.
                            If the left end is None, the interval is assumed to extend up to the right end of the previous interval.
        pira, bool: only confirm value if there are no relapses between value and confirmation
        verbose, int: 0[default, print no info], 1[print concise info], 2[print extended info]
    RETURNS:
        True if value is confirmed, False otherwise.
    """

    try:
       _ = (e for e in conf_weeks) # check if conf_weeks is iterable
    except TypeError:
       conf_weeks = [conf_weeks] # if it's not, make it a list with a single element


    if pira and (relapse_indep is None):
        relapse_indep = {'bl': (0, 0), 'event': (90, 30), 'conf': (90, 30)}

    # If more than one event happen on the same day, only keep last
    udates, ucounts = np.unique(data[date_col].values, return_counts=True)
    if any(ucounts>1):
        data = data.groupby(date_col).last().reset_index()

    data = data.reset_index(drop=True)
    nvisits = len(data)

    if relapse is not None and rdate_col is None:
        rdate_col = date_col

    # Remove missing values from columns of interest
    data = data[[value_col, date_col]].copy().dropna()
    # Convert dates to datetime.date format
    data[date_col] = pd.to_datetime(data[date_col]) #col_to_date(data[date_col]) #
    if relapse is None:
        relapse = pd.DataFrame([], columns=[rdate_col])
        relapse_start = data[date_col].min()
    else:
        relapse = relapse[[rdate_col]].copy().dropna() # remove missing values from columns of interest
        relapse[rdate_col] = pd.to_datetime(relapse[rdate_col]) #col_to_date(relapse[rdate_col]) #
        relapse_start = relapse[rdate_col].min()
    # Convert dates to days from minimum #_d_#
    global_start = min(data[date_col].min(), relapse_start)
    relapse[rdate_col] = (relapse[rdate_col] - global_start).apply(lambda x : x.days)
    data[date_col] = (data[date_col] - global_start).apply(lambda x : x.days)

    if relapse is not None:
        nrel = len(relapse)
        relapse_dates = relapse[rdate_col].values
        relapse_df = pd.DataFrame([relapse_dates]*len(data))
        relapse_df['visit'] = data[date_col].values
        dist = relapse_df.drop(['visit'],axis=1).subtract(relapse_df['visit'], axis=0) #_d_# .apply(lambda x : pd.to_timedelta(x).dt.days)
        distm = - dist.mask(dist>0, other= - float('inf'))
        distp = dist.mask(dist<0, other=float('inf'))
        data.insert(0, 'closest_rel-', float('inf') if all(distm.isna()) else distm.min(axis=1))
        data.insert(0, 'closest_rel+', float('inf') if all(distp.isna()) else distp.min(axis=1))

        # all_dates, ii = np.unique(list(data[date_col].values) + list(relapse_dates),
        #                       return_index=True) # numpy unique() returns sorted values
        # sorted_ind = np.arange(nvisits+nrel)[ii]
        # is_rel = [x in relapse_dates for x in all_dates] # whether a date is a relapse
        # # If there is a relapse with no visit, readjust the indices:
        # date_dict = {sorted_ind[i] : i for i in range(len(sorted_ind))}
    else:
        data.insert(0, 'closest_rel-', float('inf'))
        data.insert(0, 'closest_rel+', float('inf'))
    #     is_rel = [False for x in data[date_col].values]
    #     date_dict = {i : i for i in range(len(is_rel))}

    if data.loc[idx, 'closest_rel-'] < relapse_to_event:
        return False

    if min_confirmed is None:
        milestone = data.loc[idx,value_col]

    # conf_window = (int(conf_weeks*7) - conf_tol_days, float('inf') if conf_unbounded_right
    #                else int(conf_weeks*7) + conf_tol_days)
    conf_window = [(int(c*7) - conf_tol_days[0], float('inf')) if conf_unbounded_right
                       else (int(c*7) - conf_tol_days[0], int(c*7) + conf_tol_days[1]) for c in conf_weeks]

    # conf_idx = next((x for x in range(idx+1, nvisits)
    #         if conf_window[0] <= (data.loc[x,date_col] - data.loc[idx,date_col]) <= conf_window[1] #_d_# .days # date in confirmation range
    #         and data.loc[x,'closest_rel-'] > relapse_to_conf), # out of relapse influence
    #         None)
    conf_idx = [next((x for x in range(idx+1, nvisits)
                        if c[0] <= data.loc[x,date_col] - data.loc[idx,date_col] <= c[1] # date in confirmation range
                        and data.loc[x,'closest_rel-'] >= relapse_to_conf), # occurring at least `relapse_to_conf` days from last relapse
                        None) for c in conf_window]
    conf_idx = [c for c in conf_idx if c is not None]
    conf_idx = None if len(conf_idx)==0 else min(conf_idx)
    if conf_idx is not None and all([data.loc[x,value_col]
                >= min(milestone, data.loc[idx,value_col]) for x in range(idx+1,conf_idx+1)]):
        ###
        if pira:
            intervals = {ic : [] for ic in conf_idx}
            for ic in conf_idx:
                for point in ('event', 'conf'):
                    t = data.loc[idx,date_col] if point=='event' else data.loc[ic,date_col]
                    if relapse_indep[point][0] is not None:
                        t0 = t - relapse_indep[point][0]
                    if relapse_indep[point][1] is not None:
                        t1 = t + relapse_indep[point][1]
                        if t1>t0:
                            intervals[ic].append([t0,t1])
            rel_inbetween = [np.logical_or.reduce([(a[0]<=relapse_dates) & (relapse_dates<=a[1])
                            for a in intervals[ic]]).any() for ic in conf_idx]
            valid = not any(rel_inbetween) #not any(is_rel[date_dict[idx]:date_dict[conf_idx]+1])
        else:
            valid = True
        ###
        return valid
    else:
        return False
