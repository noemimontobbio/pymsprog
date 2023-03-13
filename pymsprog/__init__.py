
"""
This module defines the functions to compute MS progression.
"""

__version__ = "0.1.0"


import numpy as np
import pandas as pd

import datetime

#####################################################################################

def MSprog(data, subj_col, value_col, date_col,
           relapse=None, rsubj_col=None, rdate_col=None,
           outcome='edss', conf_months=[6,12], conf_tol=45, conf_left=False, rel_infl=30,
           event='multiple', baseline='roving', sub_threshold=False, relapse_rebl=True,
           min_value=0, include_dates=False, include_value=False, verbose=2):
    """
    Compute MS progression from longitudinal data.
    ARGUMENTS:
        data, DataFrame: longitudinal data containing subject ID, outcome value, date of visit
        subj_col, str: name of data column with subject ID
        value_col, str: name of data column with outcome value
        date_col, str: name of data column with date of visit
        relapse, DataFrame: (optional) longitudinal data containing subject ID and relapse date
        rsubj_col / rdate_col, str: name of columns for relapse data, if different from outcome data
        outcome, str: 'edss'[default],'nhpt','t25fw','sdmt'
        conf_months, int or list-like : period before confirmation (months)
        conf_tol, int: tolerance window for confirmation visit (days): [t(months)-conf_tol(days), t(months)+conf_tol(days)]
        conf_left, bool: if True, confirmation window is [t(months)-conf_tol(days), inf)
        rel_infl, int: influence of last relapse (days)
        event, str: 'first', 'multiple'[default]
        baseline, str: 'fixed', 'roving'[default]
        sub_threshold, bool: if True, include confirmed sub-threshold events for roving baseline
        relapse_rebl, bool: if True, search for PIRA events again with post-relapse re-baseline
        min_value, float: only consider progressions events where the outcome is >= value
        include_dates, bool: if True, report dates of events
        include_value, bool: if True, report value of outcome at event
        verbose, int: 0[print no info], 1[print concise info], 2[default, print extended info]
    RETURNS:
        DataFrame containing event sequence and info.
    """

    #####################################################################################
    # SETUP

    if isinstance(conf_months, int):
        conf_months = [conf_months]

    if rsubj_col is None:
        rsubj_col = subj_col
    if rdate_col is None:
        rdate_col = date_col

    # Remove missing values
    data = data.dropna()
    relapse = relapse.dropna()
    # Convert dates to datetime format
    data[date_col] = col_to_date(data[date_col])
    relapse[rdate_col] = col_to_date(relapse[rdate_col])

    # Define progression delta
    def delta(value):
        return compute_delta(value, outcome)

    #####################################################################################
    # Assess progression

    all_subj = data[subj_col].unique()
    nsub = len(all_subj)
    results = pd.DataFrame([[''] + ([0, None, None, None, None] + [None]*len(conf_months) + [None, None])*3]*nsub,
               columns=['event_sequence',
                        'improvement', 'impr_bl', 'impr_date', 'impr_value', 'time2impr'] + ['impr_conf'+str(m) for m in conf_months]
                       + ['impr_sust_days', 'impr_sust_last',
                        'progression', 'prog_bl', 'prog_date', 'prog_value', 'time2prog'] + ['prog_conf'+str(m) for m in conf_months]
                       + ['prog_sust_days', 'prog_sust_last',
                        'PIRA', 'PIRA_bl', 'PIRA_date', 'PIRA_value', 'time2PIRA'] + ['PIRA_conf'+str(m) for m in conf_months]
                       + ['PIRA_sust_days', 'PIRA_sust_last'])
    results.insert(loc=10, column='prog_type', value=[None]*nsub)
    results.insert(loc=18, column='RAW', value=[0]*nsub)

    results.index = all_subj


    for subjid in all_subj:

        data_id = data.loc[data[subj_col]==subjid,:].reset_index(drop=True)
        nvisits = len(data_id)
        first_visit = data_id[date_col].min()
        relapse_id = relapse.loc[relapse[rsubj_col]==subjid,:].reset_index(drop=True)
        relapse_id = relapse_id.loc[relapse_id[rdate_col]>=first_visit+datetime.timedelta(days=rel_infl),:] # ignore relapses occurring before first visit
        relapse_dates = relapse_id[rdate_col].values
        nrel = len(relapse_dates)

        all_dates, ii = np.unique(np.concatenate([data_id[date_col].values, relapse_dates]),
                              return_index=True) # numpy unique() returns sorted values
        sorted_ind = np.arange(nvisits+nrel)[ii]
        is_rel = [x in relapse_dates for x in all_dates] # whether a date is a relapse
        # If there is a relapse with no visit, readjust the indices:
        date_dict = {sorted_ind[i] : i for i in range(len(sorted_ind))}

        relapse_df = pd.DataFrame([relapse_dates]*len(data_id))
        relapse_df['visit'] = data_id[date_col].values
        dist = relapse_df.drop(['visit'],axis=1).subtract(relapse_df['visit'], axis=0).apply(lambda x : pd.to_timedelta(x).dt.days)
        distm = - dist.mask(dist>0, other= - float('inf'))
        distp = dist.mask(dist<0, other=float('inf'))
        data_id['closest_rel-'] = float('inf') if all(distm.isna()) else distm.min(axis=1)
        data_id['closest_rel+'] = float('inf') if all(distp.isna()) else distp.min(axis=1)


        if verbose > 0:
            print('\nSubject #%d: %d visit%s, %d relapse%s'
              %(subjid,nvisits,'' if nvisits==1 else 's',nrel,'' if nrel==1 else 's'))

        # if event == 'multiple':
        impr_bl, impr_date, impr_value, time2impr, impr_conf, impr_sustd, impr_sustl = [], [], [], [], {m : [] for m in conf_months}, [], []
        prog_bl, prog_date, prog_value, time2prog, prog_conf, prog_sustd, prog_sustl = [], [], [], [], {m : [] for m in conf_months}, [], []
        pira_bl, pira_date, pira_value, time2pira, pira_conf, pira_sustd, pira_sustl = [], [], [], [], {m : [] for m in conf_months}, [], []
        prog_type, event_sequence, event_index = [], [''], []


        bl_idx, search_idx = 0, 1 # baseline index and index of where we are in the search
        proceed = 1
        phase = 0 # if post-relapse re-baseline is enabled (relapse_rebl==True),
                  # phase will become 1 when re-searching for PIRA events
        conf_window = [(int(c*30.5) - conf_tol, float('inf')) if conf_left
                       else (int(c*30.5) - conf_tol, int(c*30.5) + conf_tol) for c in conf_months]

        while proceed:

            # Set baseline
            bl = data_id.iloc[bl_idx,:]

            # Event detection
            change_idx = next((x for x in range(search_idx,nvisits)
                        if data_id.loc[x,value_col]!=bl[value_col]), None) # first occurring value!=baseline
            if change_idx is None: # value does not change in any subsequent visit
                conf_idx = []
                proceed = 0
                if verbose == 2:
                    print('No %s change in any subsequent visit: end process' %outcome.upper())
            else:
                conf_idx = [next((x for x in range(change_idx+1, nvisits)
                        if c[0] <= (data_id.loc[x,date_col] - data_id.loc[change_idx,date_col]).days <= c[1] # date in confirmation range
                        and data_id.loc[x,'closest_rel-'] > rel_infl), # out of relapse influence
                        None) for c in conf_window]
                conf_t = [conf_months[i] for i in range(len(conf_months)) if conf_idx[i] is not None]
                conf_idx, ind = np.unique([ic for ic in conf_idx if ic is not None], return_index=True)
                conf_t = [conf_t[i] for i in ind]
                if verbose == 2:
                    print('%s change at %dth visit (%s); potential confirmation visits available: %sth'
                          %(outcome.upper(), change_idx+1 ,data_id.loc[change_idx,date_col], conf_idx+1))

                # Confirmation
                # ============

                # CONFIRMED IMPROVEMENT:
                # --------------------
                if (len(conf_idx) > 0 # confirmation visits available
                        and data_id.loc[change_idx,value_col] - bl[value_col] <= - delta(bl[value_col]) # value decreased (>delta) from baseline
                        and all([data_id.loc[x,value_col] - bl[value_col] <= - delta(bl[value_col])
                                 for x in range(change_idx+1,conf_idx[0]+1)]) # decrease is confirmed at first valid date
                        and phase == 0 # skip if re-checking for PIRA after post-relapse re-baseline
                    ):
                    next_change = next((x for x in range(conf_idx[0]+1,nvisits)
                        if data_id.loc[x,value_col] - bl[value_col] > - delta(bl[value_col])), None)
                    conf_idx = conf_idx if next_change is None else [ic for ic in conf_idx if ic<next_change] # confirmed visits
                    conf_t = conf_t[:len(conf_idx)]
                    # sustained until:
                    next_change = next((x for x in range(conf_idx[-1]+1,nvisits)
                    if data_id.loc[x,value_col] - bl[value_col] > - delta(bl[value_col]) # either decrease not sustained
                    or abs(data_id.loc[x,value_col] - data_id.loc[conf_idx[-1],value_col])
                                        >= delta(data_id.loc[conf_idx[-1],value_col]) # or further valid change from confirmation
                                    ), None)
                    next_nonsust = next((x for x in range(conf_idx[-1]+1,nvisits)
                    if data_id.loc[x,value_col] - bl[value_col] > - delta(bl[value_col]) # decrease not sustained
                        ), None)
                    sust_idx = nvisits-1 if next_nonsust is None else next_nonsust-1


                    event_sequence.append('impr')
                    event_index.append(change_idx)
                    impr_bl.append(bl[date_col].date())
                    impr_date.append(data_id.loc[change_idx,date_col].date())
                    impr_value.append(data_id.loc[change_idx,value_col])
                    time2impr.append((data_id.loc[change_idx,date_col] - bl[date_col]).days)
                    for m in conf_months:
                        impr_conf[m].append(1 if m in conf_t else 0)
                    impr_sustd.append((data_id.loc[sust_idx,date_col] - data_id.loc[conf_idx[-1],date_col]).days)
                    impr_sustl.append(int(sust_idx == nvisits-1)) #int(data_id.loc[nvisits-1,value_col] - bl[value_col] <= - delta(bl[value_col]))
                    results.loc[subjid,'improvement'] += 1

                    if baseline=='roving':
                        bl_idx = nvisits-1 if next_change is None else next_change-1 # set new baseline at last confirmation time
                        search_idx = bl_idx + 1
                    else:
                        search_idx = nvisits if next_nonsust is None else next_nonsust

                    if verbose == 2:
                        print('%s improvement (%dth visit, %s) confirmed at %s months, sustained up to %dth visit (%s)'
                              %(outcome.upper(), change_idx+1, data_id.loc[change_idx,date_col],
                                conf_t, sust_idx+1, data_id.loc[sust_idx,date_col]))
                        print('New settings: baseline at %dth visit, searching for events from %sth visit on'
                              %(bl_idx+1, '-' if search_idx>=nvisits else search_idx+1))

                # Confirmed sub-threshold improvement: RE-BASELINE
                # ------------------------------------------------
                elif (len(conf_idx) > 0 # confirmation visits available
                        and data_id.loc[change_idx,value_col]<bl[value_col] # value decreased from baseline
                        and data_id.loc[conf_idx[0],value_col]<bl[value_col] # decrease is confirmed
                        and baseline == 'roving' and sub_threshold
                        and phase == 0 # skip if re-checking for PIRA after post-relapse re-baseline
                        ):
                    next_change = next((x for x in range(conf_idx[0]+1,nvisits)
                        if data_id.loc[x,value_col]>bl[value_col]), None)
                    bl_idx = nvisits-1 if next_change is None else next_change-1 # set new baseline at last consecutive decreased value
                    search_idx = next_change
                    if verbose == 2:
                        print('Confirmed sub-threshold %s improvement (%dth visit)'
                              %(outcome.upper(), change_idx+1))
                        print('New settings: baseline at %dth visit, searching for events from %sth visit on'
                              %(bl_idx+1, '' if search_idx is None else search_idx+1))

                # CONFIRMED PROGRESSION:
                # ---------------------
                elif (len(conf_idx) > 0 # confirmation visits available
                        and data_id.loc[change_idx,value_col] >= min_value
                        and data_id.loc[change_idx,value_col] - bl[value_col] >= delta(bl[value_col]) # value increased (>delta) from baseline
                        and all([data_id.loc[x,value_col] - bl[value_col] >= delta(bl[value_col])
                                 for x in range(change_idx+1,conf_idx[0]+1)]) # increase is confirmed at first valid date
                        and all([data_id.loc[x,value_col] >= min_value for x in range(change_idx+1,conf_idx[0]+1)]) # confirmation above min_value too
                        ):
                    next_change = next((x for x in range(conf_idx[0]+1,nvisits)
                        if data_id.loc[x,value_col] - bl[value_col] < delta(bl[value_col])), None)
                    conf_idx = conf_idx if next_change is None else [ic for ic in conf_idx if ic<next_change] # confirmed dates
                    conf_t = conf_t[:len(conf_idx)]
                    # sustained until:
                    next_change = next((x for x in range(conf_idx[-1]+1,nvisits)
                        if data_id.loc[x,value_col] - bl[value_col] < delta(bl[value_col]) # either increase not sustained
                        or abs(data_id.loc[x,value_col] - data_id.loc[conf_idx[-1],value_col])
                                        >= delta(data_id.loc[conf_idx[-1],value_col]) # or further valid change from confirmation
                                    ), None)
                    next_nonsust = next((x for x in range(conf_idx[-1]+1,nvisits)
                        if data_id.loc[x,value_col] - bl[value_col] < delta(bl[value_col]) # increase not sustained
                                    ), None)
                    sust_idx = nvisits-1 if next_nonsust is None else next_nonsust-1

                    if phase == 0 and data_id.loc[change_idx,'closest_rel-'] <= rel_infl: # event occurs within relapse influence
                        prog_type.append('RAW')
                        event_sequence.append('RAW')
                        event_index.append(change_idx)
                        results.loc[subjid,'RAW'] += 1
                    elif data_id.loc[change_idx,'closest_rel-'] > rel_infl: # event occurs out of relapse influence
                        rel_inbetween = [any(is_rel[date_dict[bl_idx]:date_dict[ic]+1]) for ic in conf_idx]
                        pconf_idx = conf_idx if not any(rel_inbetween) else conf_idx[:next(i for i in range(len(conf_idx))
                                                                                          if rel_inbetween[i])]
                        if len(pconf_idx)>0 and data_id.loc[pconf_idx[-1],'closest_rel+']<=rel_infl:
                            pconf_idx = pconf_idx[:-1]
                        pconf_t = conf_t[:len(pconf_idx)]
                        if len(pconf_idx)>0:
                            prog_type.append('PIRA')
                            pira_bl.append(bl[date_col].date())
                            pira_date.append(data_id.loc[change_idx,date_col].date())
                            pira_value.append(data_id.loc[change_idx,value_col])
                            time2pira.append((data_id.loc[change_idx,date_col] - bl[date_col]).days)
                            for m in conf_months:
                                pira_conf[m].append(1 if m in pconf_t else 0)
                            pira_sustd.append((data_id.loc[sust_idx,date_col] - data_id.loc[pconf_idx[-1],date_col]).days)
                            pira_sustl.append(int(sust_idx == nvisits-1))
                            results.loc[subjid,'PIRA'] += 1
                            event_sequence.append('PIRA')
                            event_index.append(change_idx)
                        elif phase == 0:
                            event_sequence.append('prog')
                            event_index.append(change_idx)
                            prog_type.append('-')

                    if event_sequence[-1] == 'PIRA' or phase == 0:
                        prog_bl.append(bl[date_col].date())
                        prog_date.append(data_id.loc[change_idx,date_col].date())
                        prog_value.append(data_id.loc[change_idx,value_col])
                        time2prog.append((data_id.loc[change_idx,date_col] - bl[date_col]).days)
                        for m in conf_months:
                            prog_conf[m].append(1 if m in conf_t else 0)
                        prog_sustd.append((data_id.loc[sust_idx,date_col] - data_id.loc[conf_idx[-1],date_col]).days)
                        prog_sustl.append(int(sust_idx == nvisits-1))
                        results.loc[subjid,'progression'] += 1
                        if verbose == 2:
                            print('%s progression[%s] (%dth visit, %s) confirmed at %s months, sustained up to %dth visit (%s)'
                                  %(outcome.upper(), event_sequence[-1], change_idx+1, data_id.loc[change_idx,date_col],
                                    conf_t, sust_idx+1, data_id.loc[sust_idx,date_col]))


                    if baseline=='roving' or (event_sequence[-1]=='PIRA' and phase==1):
                        bl_idx = nvisits-1 if next_change is None else next_change-1 # set new baseline at last confirmation time
                        search_idx = bl_idx + 1
                    else:
                        search_idx = nvisits if next_nonsust is None else next_nonsust
                    if verbose == 2 and phase == 0:
                        print('New settings: baseline at %dth visit, searching for events from %sth visit on'
                              %(bl_idx+1, '-' if search_idx>=nvisits else search_idx+1))

                # Confirmed sub-threshold progression: RE-BASELINE
                # ------------------------------------------------
                elif (len(conf_idx) > 0 # confirmation visits available
                        and data_id.loc[change_idx,value_col]>bl[value_col] # value increased from baseline
                        and data_id.loc[conf_idx[0],value_col]>bl[value_col] # increase is confirmed
                        and baseline == 'roving' and sub_threshold
                        and phase == 0 # skip if re-checking for PIRA after post-relapse re-baseline
                        ):
                    next_change = next((x for x in range(conf_idx[0]+1,nvisits)
                        if data_id.loc[x,value_col]<bl[value_col]), None)
                    bl_idx = nvisits-1 if next_change is None else next_change-1 # set new baseline at last consecutive increased value
                    search_idx = bl_idx + 1
                    if verbose == 2:
                        print('Confirmed sub-threshold %s progression (%dth visit)'
                              %(outcome.upper(), change_idx+1))
                        print('New settings: baseline at %dth visit, searching for events from %dth visit on'
                              %(bl_idx+1, search_idx+1))

                # NO confirmation:
                # ----------------
                else:
                    search_idx = change_idx + 1 # skip the change and look for other patterns after it
                    if verbose == 2:
                        print('Change not confirmed: proceed with search')


            if phase==0 and not proceed:
                nevents = len(event_sequence) - 1

            if relapse_rebl and phase==0 and not proceed and not results.loc[subjid,'PIRA']:
                phase = 1
                proceed = 1
                search_idx = bl_idx + 1
                if verbose == 2:
                    print('Completed search with fixed baseline, re-search for PIRA events with post-relapse rebaseline')

            if proceed and (event == 'first' # only first event of each kind is considered
                and results.loc[subjid,'improvement']
                # and results.loc[subjid,'progression']
                and results.loc[subjid,'PIRA']):
                    proceed = 0
                    if verbose == 2:
                        print('%s improvement, (progression, )and PIRA events already found: end process'
                              %outcome.upper())

            if proceed and search_idx <= nvisits-2 and relapse_rebl and phase == 1:
                bl_idx = next((x for x in range(bl_idx+1,nvisits) # visits after current baseline (or after last confirmed PIRA)
                            if any(is_rel[date_dict[bl_idx]:date_dict[x]+1]) # after a relapse
                            and data_id.loc[x,'closest_rel-']>rel_infl), # out of relapse influence
                            #and data_id.loc[x,value_col] > bl[value_col]), # value worse than before the relapse
                            None)
                if bl_idx is not None:
                    search_idx = bl_idx + 1
                    if verbose == 2:
                        print('New settings: baseline at %dth visit, searching for events from %dth visit on'
                              %(bl_idx+1, search_idx+1))

            if proceed and (bl_idx is None or bl_idx > nvisits-3):
                proceed = 0
                if verbose == 2:
                    print('Not enough visits after current baseline: end process')


        event_sequence = event_sequence[1:]
        event_order = np.argsort(event_index)
        event_sequence = [event_sequence[i] for i in event_order]
        prog_index = [event_index[i] for i in range(len(event_sequence)) if event_sequence[i] in ('PIRA', 'RAW', 'prog')]
        prog_order = np.argsort(prog_index)
        if event == 'multiple':
            results.at[subjid,'impr_bl'] = None if impr_bl==[] else impr_bl
            results.at[subjid,'impr_date'] = None if impr_date==[] else impr_date
            results.at[subjid,'impr_value'] = None if impr_value==[] else impr_value
            results.at[subjid,'time2impr'] = None if time2impr==[] else time2impr
            for m in conf_months:
                results.at[subjid,'impr_conf'+str(m)] = None if impr_conf[m]==[] else impr_conf[m]
            results.at[subjid,'impr_sust_days'] = None if impr_sustd==[] else impr_sustd
            results.at[subjid,'impr_sust_last'] = None if impr_sustl==[] else impr_sustl
            results.at[subjid,'prog_bl'] = None if prog_bl==[] else [prog_bl[i] for i in prog_order]
            results.at[subjid,'prog_date'] = None if prog_date==[] else [prog_date[i] for i in prog_order]
            results.at[subjid,'prog_value'] = None if prog_value==[] else [prog_value[i] for i in prog_order]
            results.at[subjid,'time2prog'] = None if time2prog==[] else [time2prog[i] for i in prog_order]
            for m in conf_months:
                results.at[subjid,'prog_conf'+str(m)] = None if prog_conf[m]==[] else [prog_conf[m][i] for i in prog_order]
            results.at[subjid,'prog_sust_days'] = None if prog_sustd==[] else [prog_sustd[i] for i in prog_order]
            results.at[subjid,'prog_sust_last'] = None if prog_sustl==[] else [prog_sustl[i] for i in prog_order]
            results.at[subjid,'prog_type'] = None if prog_type==[] else [prog_type[i] for i in prog_order]
            results.at[subjid,'PIRA_bl'] = None if pira_bl==[] else pira_bl
            results.at[subjid,'PIRA_date'] = None if pira_date==[] else pira_date
            results.at[subjid,'PIRA_value'] = None if pira_value==[] else pira_value
            results.at[subjid,'time2PIRA'] = None if time2pira==[] else time2pira
            results.at[subjid,'PIRA_sust_days'] = None if pira_sustd==[] else pira_sustd
            results.at[subjid,'PIRA_sust_last'] = None if pira_sustl==[] else pira_sustl
            for m in conf_months:
                results.at[subjid,'PIRA_conf'+str(m)] = None if pira_conf[m]==[] else pira_conf[m]
        else:
            results.at[subjid,'improvement'] = int(results.at[subjid,'improvement']>0)
            results.at[subjid,'progression'] = int(results.at[subjid,'progression']>0)
            results.at[subjid,'PIRA'] = int(results.at[subjid,'PIRA']>0)
            results.at[subjid,'RAW'] = int(results.at[subjid,'RAW']>0)
            results.at[subjid,'impr_bl'] = None if impr_bl==[] else impr_bl[0]
            results.at[subjid,'impr_date'] = None if impr_date==[] else impr_date[0]
            results.at[subjid,'impr_value'] = None if impr_value==[] else impr_value[0]
            results.at[subjid,'time2impr'] = None if time2impr==[] else time2impr[0]
            for m in conf_months:
                results.at[subjid,'impr_conf'+str(m)] = None if impr_conf[m]==[] else impr_conf[m][0]
            results.at[subjid,'impr_sust_days'] = None if impr_sustd==[] else impr_sustd[0]
            results.at[subjid,'impr_sust_last'] = None if impr_sustl==[] else impr_sustl[0]
            first_pira = next((x for x in range(len(prog_type)) if prog_type[x]=='PIRA'), None)
            i_tmp = 0 if first_pira is None else first_pira
            results.at[subjid,'prog_bl'] = None if prog_bl==[] else prog_bl[i_tmp]
            results.at[subjid,'prog_date'] = None if prog_date==[] else prog_date[i_tmp]
            results.at[subjid,'prog_value'] = None if prog_value==[] else prog_value[i_tmp]
            results.at[subjid,'time2prog'] = None if time2prog==[] else time2prog[i_tmp]
            for m in conf_months:
                results.at[subjid,'prog_conf'+str(m)] = None if prog_conf[m]==[] else prog_conf[m][i_tmp]
            results.at[subjid,'prog_sust_days'] = None if prog_sustd==[] else prog_sustd[i_tmp]
            results.at[subjid,'prog_sust_last'] = None if prog_sustl==[] else prog_sustl[i_tmp]
            results.at[subjid,'prog_type'] = None if prog_type==[] else prog_type[i_tmp]
            results.at[subjid,'PIRA_bl'] = None if pira_bl==[] else pira_bl[0]
            results.at[subjid,'PIRA_date'] = None if pira_date==[] else pira_date[0]
            results.at[subjid,'PIRA_value'] = None if pira_value==[] else pira_value[0]
            results.at[subjid,'time2PIRA'] = None if time2pira==[] else time2pira[0]
            results.at[subjid,'PIRA_sust_days'] = None if pira_sustd==[] else pira_sustd[0]
            results.at[subjid,'PIRA_sust_last'] = None if pira_sustl==[] else pira_sustl[0]
            for m in conf_months:
                results.at[subjid,'PIRA_conf'+str(m)] = None if pira_conf[m]==[] else pira_conf[m][0]
            impr_idx = next((x for x in range(len(event_sequence)) if event_sequence[x]=='impr'), None)
            prog_idx = next((x for x in range(len(event_sequence)) if event_sequence[x] in ('RAW', 'prog')), None)
            pira_idx = next((x for x in range(len(event_sequence)) if event_sequence[x]=='PIRA'), None)
            first_events = [impr_idx, prog_idx] if pira_idx is None else [impr_idx, pira_idx]
            first_events = [ii for ii in first_events if ii is not None]
            first_events.sort()
            event_sequence = [event_sequence[ii] for ii in first_events]
        results.at[subjid,'event_sequence'] = ', '.join(event_sequence)

        if verbose > 0:
            print('Event sequence: %s'
              %(results.loc[subjid,'event_sequence'] if results.loc[subjid,'event_sequence']!='' else '-'))

    if verbose>=1:
        print('\n---\nOutcome: %s\nConfirmation at: %s%s mm (+-%ddd)\nBaseline: %s%s%s\nEvents detected: %s of each kind'
          %(outcome.upper(), 'AT LEAST ' if conf_left else '', conf_months, conf_tol,
            baseline, ' (sub-threshold)' if sub_threshold else '',
            ' + post-relapse re-baseline' if relapse_rebl else '', event))
        print('---\nTotal subjects: %d\nImprovement: %d\nProgression: %d (PIRA: %d; RAW: %d)'
              %(nsub, (results['improvement']>0).sum(), (results['progression']>0).sum(),
                (results['PIRA']>0).sum(), (results['RAW']>0).sum()))
        if min_value > 0:
            print('---\n*** WARNING only progressions to EDSS>=%d ***' %min_value)

    columns = results.columns
    if not include_dates:
        columns = [c for c in columns if not c.endswith('date')]
    if  not include_value:
        columns = [c for c in columns if not c.endswith('value')]
    results = results[columns]

    return results





#####################################################################################

def col_to_date(column, format=None, remove_na=False):
    """
    Convert dataframe column into datetime.date format.
    Arguments:
     column: the dataframe column to convert
     format: date format (see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)
    Returns:
     minimum delta corresponding to valid change
    """
    vtype = np.vectorize(lambda x: type(x))

    column_all = column.copy()
    naidx = [] if remove_na else column.isna()
    column.dropna(inplace=True)

    if all([d is pd.Timestamp for d in vtype(column)]):
        dates = column.dt.date
    elif all([d is datetime.datetime for d in vtype(column)]):
        dates = column.apply(lambda x: x.date())
    elif not all([d is datetime.date for d in vtype(column)]):
        dates = pd.to_datetime(column, format=format).dt.date
    else:
        dates = column

    column_all.loc[~naidx] = dates
    return column_all


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
    date, dob = col_to_date(date, remove_na=remove_na), col_to_date(dob, remove_na=remove_na)

    diff = pd.Series(np.nan, index=date.index, name=col_name)

    naidx = (date.isna()) | (dob.isna())
    date, dob = date.loc[~naidx].copy(), dob.loc[~naidx].copy()

    dob.loc[(dob.apply(lambda x: x.day)==29)
            & (dob.apply(lambda x: x.month)==2)] = dob.loc[(dob.apply(lambda x: x.day)==29)
            & (dob.apply(lambda x: x.month)==2)].apply(lambda dt: dt.replace(day=28))

    # bd = []
    # for x in date.index:
    #     try:
    #         bd.append(datetime.date(date[x].year, dob[x].month, dob[x].day))
    #     except ValueError:
    #         bd.append(datetime.date(date[x].year, dob[x].month, dob[x].day-1))
    # this_year_birthday = pd.Series(bd, index=date.index)

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
            raise ValueError('invalid EDSS baseline')
    elif outcome in ('nhpt', 't25fw'):
        return baseline/5
    elif outcome == 'sdmt':
        return min(baseline/10, 3)

#####################################################################################

