
"""
This module defines the functions to compute MS progression.
"""

__version__ = "0.1.1"


import numpy as np
import pandas as pd

import datetime

#import warnings

#####################################################################################

def MSprog(data, subj_col, value_col, date_col, subjects=None,
           relapse=None, rsubj_col=None, rdate_col=None, outcome='edss', delta_fun=None,
           conf_months=3, conf_tol_days=30, conf_left=False, require_sust_months=0, rel_infl=30,
           event='firstprog', baseline='fixed', sub_threshold=False, relapse_rebl=False,
           min_value=0, prog_last_visit=False, include_dates=False, include_value=False, verbose=1):
    """
    Compute MS progression from longitudinal data.
    ARGUMENTS:
        data, DataFrame: longitudinal data containing subject ID, outcome value, date of visit
        subj_col, str: name of data column with subject ID
        value_col, str: name of data column with outcome value
        date_col, str: name of data column with date of visit
        subjects, list-like : (optional) subset of subjects
        relapse, DataFrame: (optional) longitudinal data containing subject ID and relapse date
        rsubj_col / rdate_col, str: name of columns for relapse data, if different from outcome data
        outcome, str: 'edss'[default],'nhpt','t25fw','sdmt'
        delta_fun, function: (optional) Custom function specifying the minimum delta corresponding to a valid change from baseline.
        conf_months, int or list-like : period before confirmation (months)
        conf_tol_days, int or list-like of length 1 or 2: tolerance window for confirmation visit (days): [t(months)-conf_tol_days[0](days), t(months)+conf_tol_days[0](days)]
        conf_left, bool: if True, confirmation window is [t(months)-conf_tol_days, inf)
        require_sust_months, int: count an event as such only if sustained for _ months from confirmation
        rel_infl, int: influence of last relapse (days)
        event, str: 'first' [only the very first event - improvement or progression]
                    'firsteach' [first improvement and first progression]
                    'firstprog' [first progression]
                    'firstprogtype' [first progression of each kind - PIRA, RAW, undefined]
                    'multiple'[all events, default]
        baseline, str: 'fixed', 'roving'[default]
        sub_threshold, bool: if True, include confirmed sub-threshold events for roving baseline
        relapse_rebl, bool: if True, search for PIRA events again with post-relapse re-baseline
        min_value, float: only consider progressions events where the outcome is >= value
        prog_last_visit, bool: if True, include progressions occurring at last visit (i.e. with no confirmation)
        include_dates, bool: if True, report dates of events
        include_value, bool: if True, report value of outcome at event
        verbose, int: 0[print no info], 1[print concise info], 2[default, print extended info]
    RETURNS:
        Two DataFrames:
         - summary of detected events for each subject;
         - extended info on event sequence for each subject.
    """

    #####################################################################################
    # SETUP

    if isinstance(conf_months, int):
        conf_months = [conf_months]

    if isinstance(conf_tol_days, int):
        conf_tol_days = [conf_tol_days, conf_tol_days]

    if rsubj_col is None:
        rsubj_col = subj_col
    if rdate_col is None:
        rdate_col = date_col

    # Remove missing values from columns of interest
    data = data[[subj_col, value_col, date_col]].dropna()
    # Convert dates to datetime format
    data[date_col] = col_to_date(data[date_col])
    if relapse is None:
        relapse_rebl = False
        relapse = pd.DataFrame([], columns=[rsubj_col, rdate_col])
    else:
        relapse = relapse[[rsubj_col, rdate_col]].dropna()
        relapse[rdate_col] = col_to_date(relapse[rdate_col])

    if subjects is not None:
        data = data[data[subj_col].isin(subjects)]
        relapse = relapse[relapse[rsubj_col].isin(subjects)]

    # Define progression delta
    def delta(value):
        return compute_delta(value, outcome) if delta_fun is None else delta_fun(value)

    #####################################################################################
    # Assess progression

    all_subj = data[subj_col].unique()
    nsub = len(all_subj)
    max_nevents = round(data.groupby(subj_col)[value_col].count().max()/2)
    results = pd.DataFrame([[None]*8 + [None]*len(conf_months)
                            + [None]*(len(conf_months)-1) + [None]*2]*nsub*max_nevents,
               columns=[subj_col, 'nevent', 'event_type', 'bldate', 'blvalue', 'date', 'value', 'time2event']
                       + ['conf'+str(m) for m in conf_months]+ ['PIRA_conf'+str(m) for m in conf_months[1:]]
                       + ['sust_days', 'sust_last'])
    results[subj_col] = np.repeat(all_subj, max_nevents)
    results['nevent'] = np.tile(np.arange(1,max_nevents+1), nsub)
    summary = pd.DataFrame([[0]*6]*nsub, columns=['event_sequence', 'improvement', 'progression',
                                                  'RAW', 'PIRA', 'undefined_prog'], index=all_subj)

    for subjid in all_subj:

        data_id = data.loc[data[subj_col]==subjid,:].copy()

        # If more than one visit occur on the same day, only keep last
        udates, ucounts = np.unique(data_id[date_col].values, return_counts=True)
        if any(ucounts>1):
            data_id = data_id.groupby(date_col).last()

        # Sort visits in chronological order
        sorted_tmp = data_id.sort_values(by=[date_col])
        if any(sorted_tmp.index != data_id.index):
            data_id = sorted_tmp.copy()

        data_id.reset_index(inplace=True, drop=True)

        nvisits = len(data_id)
        first_visit = data_id[date_col].min()
        relapse_id = relapse.loc[relapse[rsubj_col]==subjid,:].copy().reset_index(drop=True)
        relapse_id = relapse_id.loc[relapse_id[rdate_col]>=first_visit-datetime.timedelta(days=rel_infl),:] # ignore relapses occurring before first visit
        relapse_dates = relapse_id[rdate_col].values
        nrel = len(relapse_dates)


        if verbose == 2:
            print('\nSubject #%s: %d visit%s, %d relapse%s'
              %(subjid, nvisits,'' if nvisits==1 else 's', nrel, '' if nrel==1 else 's'))
            if any(ucounts>1):
                print('Found multiple visits on the same day: only keeping last.')
            if any(sorted_tmp.index != data_id.index):
                print('Visits not listed in chronological order: sorting them.')


        all_dates, sorted_ind = np.unique(list(data_id[date_col].values) + list(relapse_dates), #np.concatenate([data_id[date_col].values, relapse_dates]),
                              return_index=True) # numpy unique() returns sorted values
        #sorted_ind = np.arange(nvisits+nrel)[ii]
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

        event_type, event_index = [''], []
        bldate, blvalue, edate, evalue, time2event = [], [], [], [], []
        conf, sustd, sustl = {m : [] for m in conf_months}, [], []
        pira_conf = {m : [] for m in conf_months[1:]}


        bl_idx, search_idx = 0, 1 # baseline index and index of where we are in the search
        proceed = 1
        phase = 0 # if post-relapse re-baseline is enabled (relapse_rebl==True),
                  # phase will become 1 when re-searching for PIRA events
        conf_window = [(int(c*30.44) - conf_tol_days[0], float('inf')) if conf_left
                       else (int(c*30.44) - conf_tol_days[0], int(c*30.44) + conf_tol_days[1]) for c in conf_months]

        while proceed:

            # Set baseline (skip if within relapse influence)
            while proceed and data_id.loc[bl_idx,'closest_rel-'] <= rel_infl:
                if verbose==2:
                    print('Baseline (visit no.%d) is within relapse influence: moved to visit no.%d'
                              %(bl_idx+1, bl_idx+2))
                bl_idx += 1
                search_idx += 1
                if bl_idx > nvisits-2:
                    proceed = 0
                    if verbose == 2:
                        print('Not enough visits left: end process')

            # if bl_idx > nvisits - 2:
            #     break
            if bl_idx > nvisits - 1:
                bl_idx = nvisits - 1
                proceed = 0
                if verbose == 2:
                    print('Not enough visits left: end process')

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
                conf_idx = [ic for ic in conf_idx if ic is not None]
                # conf_idx, ind = np.unique([ic for ic in conf_idx if ic is not None], return_index=True)
                # conf_t = [conf_t[i] for i in ind]
                if verbose == 2:
                    print('%s change at visit no.%d (%s); potential confirmation visits available: no.%s'
                          %(outcome.upper(), change_idx+1 ,data_id.loc[change_idx,date_col].date(), [i+1 for i in conf_idx]))

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

                    valid_impr = 1
                    if require_sust_months:
                        valid_impr = (next_nonsust is None) or (data_id.loc[next_nonsust,date_col]
                                    - data_id.loc[conf_idx[-1],date_col]).days > require_sust_months*30.44
                    if valid_impr:
                        sust_idx = nvisits-1 if next_nonsust is None else next_nonsust-1

                        event_type.append('impr')
                        event_index.append(change_idx)
                        bldate.append(bl[date_col].date())
                        blvalue.append(bl[value_col])
                        edate.append(data_id.loc[change_idx,date_col].date())
                        evalue.append(data_id.loc[change_idx,value_col])
                        time2event.append((data_id.loc[change_idx,date_col] - bl[date_col]).days)
                        for m in conf_months:
                            conf[m].append(1 if m in conf_t else 0)
                        for m in conf_months[1:]:
                            pira_conf[m].append(None)
                        sustd.append((data_id.loc[sust_idx,date_col] - data_id.loc[conf_idx[-1],date_col]).days)
                        sustl.append(int(sust_idx == nvisits-1)) #int(data_id.loc[nvisits-1,value_col] - bl[value_col] <= - delta(bl[value_col]))

                        if baseline=='roving':
                            bl_idx = nvisits-1 if next_change is None else next_change-1 # set new baseline at last confirmation time
                            search_idx = bl_idx + 1
                        else:
                            search_idx = nvisits if next_change is None else next_change #next_nonsust

                        if verbose == 2:
                            print('%s improvement (visit no.%d, %s) confirmed at %s months, sustained up to visit no.%d (%s)'
                                  %(outcome.upper(), change_idx+1, data_id.loc[change_idx,date_col].date(),
                                    conf_t, sust_idx+1, data_id.loc[sust_idx,date_col].date()))
                            print('New settings: baseline at visit no.%d, searching for events from visit no.%s on'
                                  %(bl_idx+1, '-' if search_idx>=nvisits else search_idx+1))

                    else:
                        search_idx = change_idx + 1 # skip the change and look for other patterns after it
                        if verbose == 2:
                            print('Change confirmed but not sustained for >=%d months: proceed with search'
                                  %require_sust_months)

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
                    search_idx = bl_idx + 1
                    if verbose == 2:
                        print('Confirmed sub-threshold %s improvement (visit no.%d)'
                              %(outcome.upper(), change_idx+1))
                        print('New settings: baseline at visit no.%d, searching for events from visit no.%s on'
                              %(bl_idx+1, '-' if search_idx is None else search_idx+1))

                # CONFIRMED PROGRESSION:
                # ---------------------
                elif (data_id.loc[change_idx,value_col] >= min_value
                        and data_id.loc[change_idx,value_col] - bl[value_col] >= delta(bl[value_col]) # value increased (>delta) from baseline
                    and ((len(conf_idx) > 0 # confirmation visits available
                        and all([data_id.loc[x,value_col] - bl[value_col] >= delta(bl[value_col])
                                 for x in range(change_idx+1,conf_idx[0]+1)]) # increase is confirmed at first valid date
                        and all([data_id.loc[x,value_col] >= min_value for x in range(change_idx+1,conf_idx[0]+1)]) # confirmation above min_value too
                        ) or (prog_last_visit and change_idx==nvisits-1))
                      ):
                    if change_idx==nvisits-1:
                        conf_idx = [nvisits-1]
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
                    valid_prog = 1
                    if require_sust_months:
                        valid_prog = (next_nonsust is None) or (data_id.loc[next_nonsust,date_col]
                                    - data_id.loc[conf_idx[-1],date_col]).days > require_sust_months*30.44
                    if valid_prog:
                        sust_idx = nvisits-1 if next_nonsust is None else next_nonsust-1

                        if phase == 0 and data_id.loc[change_idx,'closest_rel-'] <= rel_infl: # event occurs within relapse influence
                            event_type.append('RAW')
                            event_index.append(change_idx)
                        elif data_id.loc[change_idx,'closest_rel-'] > rel_infl: # event occurs out of relapse influence
                            rel_inbetween = [any(is_rel[date_dict[bl_idx]:date_dict[ic]+1]) for ic in conf_idx]
                            pconf_idx = conf_idx if not any(rel_inbetween) else conf_idx[:next(i for i in range(len(conf_idx))
                                                                                              if rel_inbetween[i])]
                            if len(pconf_idx)>0 and data_id.loc[pconf_idx[-1],'closest_rel+']<=rel_infl:
                                pconf_idx = pconf_idx[:-1]
                            pconf_t = conf_t[:len(pconf_idx)]
                            if len(pconf_idx)>0: # if pira is confirmed
                                for m in conf_months[1:]:
                                    pira_conf[m].append(int(m in pconf_t))
                                event_type.append('PIRA')
                                event_index.append(change_idx)
                            elif phase == 0: # if pira is not confirmed, and we're not re-searching for pira events only
                                event_type.append('prog')
                                event_index.append(change_idx)

                        if phase == 0 and event_type[-1] != 'PIRA':
                            for m in conf_months[1:]:
                                pira_conf[m].append(None)

                        if event_type[-1] == 'PIRA' or phase == 0:
                            bldate.append(bl[date_col].date())
                            blvalue.append(bl[value_col])
                            edate.append(data_id.loc[change_idx,date_col].date())
                            evalue.append(data_id.loc[change_idx,value_col])
                            time2event.append((data_id.loc[change_idx,date_col] - bl[date_col]).days)
                            for m in conf_months:
                                conf[m].append(1 if m in conf_t else 0)
                            sustd.append((data_id.loc[sust_idx,date_col] - data_id.loc[conf_idx[-1],date_col]).days)
                            sustl.append(int(sust_idx == nvisits-1))
                            if verbose == 2:
                                print('%s progression[%s] (visit no.%d, %s) confirmed at %s months, sustained up to visit no.%d (%s)'
                                      %(outcome.upper(), event_type[-1], change_idx+1, data_id.loc[change_idx,date_col].date(),
                                        conf_t, sust_idx+1, data_id.loc[sust_idx,date_col].date()))


                        if (baseline=='roving' and phase==0): #or (event_type[-1]=='PIRA' and phase==1): #
                            bl_idx = nvisits-1 if next_change is None else next_change-1 # set new baseline at last confirmation time
                            search_idx = bl_idx + 1
                        else:
                            search_idx = nvisits if next_change is None else next_change #next_nonsust
                        if verbose == 2 and phase == 0:
                            print('New settings: baseline at visit no.%d, searching for events from visit no.%s on'
                                  %(bl_idx+1, '-' if search_idx>=nvisits else search_idx+1))
                    else:
                        search_idx = change_idx + 1 # skip the change and look for other patterns after it
                        if verbose == 2:
                            print('Change confirmed but not sustained for >=%d months: proceed with search'
                                  %require_sust_months)

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

            if relapse_rebl and phase==0 and not proceed: # and 'PIRA' not in event_type:
                phase = 1
                proceed = 1
                bl_idx = 0
                search_idx = 1 #bl_idx + 1 #
                if verbose == 2:
                    print('Completed search with fixed baseline, re-search for PIRA events with post-relapse rebaseline')

            if proceed and (
                (event == 'first' and len(event_type)>1)
                or (event == 'firsteach' and ('impr' in event_type) and ('prog' in event_type))
                or (event == 'firstprog' and (('RAW' in event_type) or ('PIRA' in event_type) or ('prog' in event_type)))
                or (event == 'firstprogtype' and ('RAW' in event_type) and ('PIRA' in event_type) and ('prog' in event_type))
                        ):
                    proceed = 0
                    if verbose == 2:
                        print('First events already found: end process')

            if (proceed and search_idx <= nvisits-1 and relapse_rebl and phase == 1
                    and any(is_rel[date_dict[bl_idx]:date_dict[search_idx]+1])):
                bl_idx = next((x for x in range(bl_idx+1,nvisits) # visits after current baseline (or after last confirmed PIRA)
                            if any(is_rel[date_dict[bl_idx]:date_dict[x]+1]) # after a relapse
                            and data_id.loc[x,'closest_rel-'] > rel_infl), # out of relapse influence
                            #and data_id.loc[x,value_col] > bl[value_col]), # value worse than before the relapse
                            None)
                if bl_idx is not None:
                    search_idx = bl_idx + 1
                    if verbose == 2:
                        print('New settings: baseline at visit no.%d, searching for events from visit no.%d on'
                              %(bl_idx+1, search_idx+1))

                if proceed and (bl_idx is None or bl_idx > nvisits-2):
                    proceed = 0
                    if verbose == 2:
                        print('Not enough visits after current baseline: end process')


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
                event_index[all_ind[:-1]] = -1 # mark duplicate events with -1

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
                first_events = [0] if event=='first' else np.unique([
                    ii for ii in first_events if ii is not None]) # np.unique() returns the values already sorted
                event_type = [event_type[ii] for ii in first_events]
                event_order = [event_order[ii] for ii in first_events]

            results.drop(subj_index[len(event_type):], inplace=True)

            results.loc[results[subj_col]==subjid, 'event_type'] = event_type
            results.loc[results[subj_col]==subjid, 'bldate'] = [bldate[i] for i in event_order]
            results.loc[results[subj_col]==subjid, 'blvalue'] = [blvalue[i] for i in event_order]
            results.loc[results[subj_col]==subjid, 'date'] = [edate[i] for i in event_order]
            results.loc[results[subj_col]==subjid, 'value'] = [evalue[i] for i in event_order]
            results.loc[results[subj_col]==subjid, 'time2event'] = [time2event[i] for i in event_order]
            for m in conf_months:
                results.loc[results[subj_col]==subjid, 'conf'+str(m)] = [conf[m][i] for i in event_order]
            results.loc[results[subj_col]==subjid, 'sust_days'] = [sustd[i] for i in event_order]
            results.loc[results[subj_col]==subjid, 'sust_last'] = [sustl[i] for i in event_order]
            for m in conf_months[1:]:
                results.loc[results[subj_col]==subjid, 'PIRA_conf'+str(m)] = [pira_conf[m][i] for i in event_order]
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
        print('\n---\nOutcome: %s\nConfirmation at: %smm (-%ddd, +%s)\nBaseline: %s%s%s\nRelapse influence: %ddd\nEvents detected: %s'
          %(outcome.upper(), conf_months, conf_tol_days[0], 'inf' if conf_left else str(conf_tol_days[1])+'dd',
            baseline, ' (sub-threshold)' if sub_threshold else '',
            ' (and post-relapse re-baseline)' if relapse_rebl else '', rel_infl, event))
        print('---\nTotal subjects: %d\n---\nProgressed: %d (PIRA: %d; RAW: %d)'
              %(nsub, (summary['progression']>0).sum(),
                (summary['PIRA']>0).sum(), (summary['RAW']>0).sum()))
        if not event.startswith('firstprog'):
            print('Improved: %d' %(summary['improvement']>0).sum())
        if event in ('multiple', 'firstprogtype'):
            print('---\nProgression events: %d (PIRA: %d; RAW: %d)'
                  %(summary['progression'].sum(),
                    summary['PIRA'].sum(), summary['RAW'].sum()))
        if event=='multiple':
            print('Improved: %d' %(summary['improvement'].sum()))

        if min_value > 0:
            print('---\n*** WARNING only progressions to %s>=%d are considered ***'
                  %(outcome.upper(), min_value))

    columns = results.columns
    if not include_dates:
        columns = [c for c in columns if not c.endswith('date')]
    if not include_value:
        columns = [c for c in columns if not c.endswith('value')]
    results = results[columns]

    return summary, results.reset_index(drop=True)


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
    naidx = pd.Series([False]*len(column), index=column.index) if remove_na else column.isna()
    column.dropna(inplace=True)

    if all([d is pd.Timestamp for d in vtype(column)]):
        dates = column.dt.date
    elif all([d is datetime.datetime for d in vtype(column)]):
        dates = column.apply(lambda x : x.date())
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
    date = col_to_date(date, remove_na=remove_na)
    dob = col_to_date(dob, remove_na=remove_na)

    diff = pd.Series(np.nan, index=date.index, name=col_name)

    naidx = (date.isna()) | (dob.isna())
    date, dob = date.loc[~naidx].copy(), dob.loc[~naidx].copy()

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
            raise ValueError('invalid EDSS baseline')
    elif outcome in ('nhpt', 't25fw'):
        return baseline/5
    elif outcome == 'sdmt':
        return min(baseline/10, 3)

#####################################################################################

def value_milestone(data, milestone, value_col, date_col, subj_col,
                   relapse=None, rsubj_col=None, rdate_col=None,
                   conf_months=6, conf_tol_days=45, conf_left=False, rel_infl=30,
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
        conf_months, int or list-like : period before confirmation (months)
        conf_tol_days, int or list-like of length 1 or 2: tolerance window for confirmation visit (days): [t(months)-conf_tol[0](days), t(months)+conf_tol[0](days)]
        conf_left, bool: if True, confirmation window is [t(months)-conf_tol(days), inf)
        rel_infl, int: influence of last relapse (days)
        verbose, int: 0[default, print no info], 1[print concise info], 2[print extended info]
    RETURNS:
        DataFrame containing:
         - date of first reaching value >=milestone (or last date of follow-up if milestone is not reached);
         - first value >=milestone, if present, otherwise last value recorded
    """

    if relapse is not None and rsubj_col is None:
        rsubj_col = subj_col
    if relapse is not None and rdate_col is None:
        rdate_col = date_col

    conf_window = (int(conf_months*30.5) - conf_tol_days, float('inf') if conf_left
                   else int(conf_months*30.5) + conf_tol_days)

    all_subj = data[subj_col].unique()
    nsub = len(all_subj)
    results = pd.DataFrame([[None, None]]*nsub, columns=[date_col, value_col], index=all_subj)

    for subjid in all_subj:
        data_id = data.loc[data[subj_col]==subjid,:].copy()

        udates, ucounts = np.unique(data_id[date_col].values, return_counts=True)
        if any(ucounts>1):
            data_id = data_id.groupby(date_col).last()

        data_id.reset_index(inplace=True, drop=True)

        nvisits = len(data_id)
        if verbose > 0:
            print('\nSubject #%d: %d visit%s'
              %(subjid,nvisits,'' if nvisits==1 else 's'))
            if any(ucounts>1):
                print('Found multiple visits in the same day: only keeping last')
        first_visit = data_id[date_col].min()
        if relapse is not None:
            relapse_id = relapse.loc[relapse[rsubj_col]==subjid,:].reset_index(drop=True)
            relapse_id = relapse_id.loc[relapse_id[rdate_col]>=first_visit+datetime.timedelta(days=rel_infl),:] # ignore relapses occurring before first visit
            relapse_dates = relapse_id[rdate_col].values
            relapse_df = pd.DataFrame([relapse_dates]*len(data_id))
            relapse_df['visit'] = data_id[date_col].values
            dist = relapse_df.drop(['visit'],axis=1).subtract(relapse_df['visit'], axis=0).apply(lambda x : pd.to_timedelta(x).dt.days)
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
                    if data_id.loc[x,value_col]>=milestone), None) # first occurring value!=baseline
            if milestone_idx is None: # value does not change in any subsequent visit
                results.at[subjid,date_col] = data_id.iloc[-1,:][date_col].date()
                results.at[subjid,value_col] = data_id.iloc[-1,:][value_col]
                proceed = 0
                if verbose == 2:
                    print('No value >=%d in any visit: end process' %(milestone))
            else:
                conf_idx = next((x for x in range(milestone_idx+1, nvisits)
                        if conf_window[0] <= (data_id.loc[x,date_col] - data_id.loc[milestone_idx,date_col]).days <= conf_window[1] # date in confirmation range
                        and data_id.loc[x,'closest_rel-'] > rel_infl), # out of relapse influence
                        None)
                if conf_idx is not None and all([data_id.loc[x,value_col]
                            >= milestone for x in range(milestone_idx+1,conf_idx+1)]):
                    results.at[subjid,date_col] = data_id.loc[milestone_idx,date_col].date()
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
            results.at[subjid,date_col] = data_id.iloc[-1,:][date_col].date()
            results.at[subjid,value_col] = data_id.iloc[-1,:][value_col]

    return results


#####################################################################################


def separate_ri_ra(data, relapse, mode, value_col, date_col, subj_col,
                   rsubj_col=None, rdate_col=None, rel_infl=30, bl_rel_infl=None,
                   conf_months=6, conf_tol_days=45, conf_left=False, require_sust_months=0,
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
        rel_infl, int: influence of last relapse (days)
        bl_rel_infl, int: minimum days from last relapse to consider visit as baseline (otherwise move baseline to next visit)
        conf_months, int: period before confirmation (months)
        conf_tol_days, int or list-like of length 1 or 2: tolerance window for confirmation visit (days): [t(months)-conf_tol[0](days), t(months)+conf_tol[0](days)]
        conf_left, bool: if True, confirmation window is [t(months)-conf_tol(days), inf)
        require_sust_months, int: count an event as such only if sustained for _ months from confirmation
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

    if rsubj_col is None:
        rsubj_col = subj_col
    if rdate_col is None:
        rdate_col = date_col
    if bl_rel_infl is None:
        bl_rel_infl = rel_infl
    if isinstance(conf_tol_days, int):
        conf_tol_days = [conf_tol_days, conf_tol_days]

    data_sep = data.copy()
    ri_col, ra_col = 'ri'+value_col, 'ra'+value_col
    if mode!='none':
        data_sep[ra_col] = 0
    if mode in ('ri', 'both'):
        data_sep[ri_col] = data_sep[value_col]
    if return_rel_num:
        data_sep['relapse_num'] = 0
    if subtract_bl:
        data_sep[value_col+'-bl'] = 0
        if mode in ('ri', 'both'):
            data_sep[ri_col+'-bl'] = 0

    def delta(value):
        return compute_delta(value, 'edss')

    conf_window = (int(conf_months*30.44) - conf_tol_days[0],
                   float('inf')) if conf_left else (int(conf_months*30.44) - conf_tol_days[0],
                                                    int(conf_months*30.44) + conf_tol_days[1])

    all_subj = data[subj_col].unique()
    nsub = len(all_subj)

    if return_raw_dates:
        raw_events = []

    for subjid in all_subj:
        data_id = data_sep.loc[data_sep[subj_col]==subjid,:].copy().reset_index(drop=True)
        nvisits = len(data_id)

        relapse_id = relapse.loc[relapse[rsubj_col]==subjid,:].reset_index(drop=True)
        nrel = len(relapse_id)

        if verbose > 0:
            print('\nSubject #%d: %d visit%s, %d relapse%s'
              %(subjid,nvisits,'' if nvisits==1 else 's', nrel,'' if nrel==1 else 's'))

        relapse_dates = relapse_id[rdate_col].values
        relapse_df = pd.DataFrame([relapse_dates]*len(data_id))
        relapse_df['visit'] = data_id[date_col].values
        dist = relapse_df.drop(['visit'],axis=1).subtract(relapse_df['visit'], axis=0).apply(lambda x : pd.to_timedelta(x).dt.days)
        distm = - dist.mask(dist>0, other= - float('inf'))
        distp = dist.mask(dist<0, other=float('inf'))
        data_id['closest_rel-'] = float('inf') if all(distm.isna()) else distm.min(axis=1)
        data_id['closest_rel+'] = float('inf') if all(distp.isna()) else distp.min(axis=1)

        # First visit out of relapse influence
        rel_free_bl = next((x for x in range(len(data_id))
                        if data_id.loc[x,'closest_rel-']>bl_rel_infl), None)
        if rel_free_bl is None:
            data_id = data_id.loc[[],:].reset_index(drop=True)
            if verbose==2:
                print('No baseline visits out of relapse influence')
        elif rel_free_bl>0:
            data_id = data_id.loc[rel_free_bl:,:].reset_index(drop=True)
            if verbose==2:
                print('Moving baseline to first visit out of relapse influence (%dth)' %(rel_free_bl+1))
        nvisits = len(data_id)


        first_visit = data_id[date_col].min()
        relapse_id = relapse_id.loc[relapse_id[rdate_col]
                        >= first_visit, #- datetime.timedelta(days=bl_rel_infl), # ignore relapses occurring before or at first visit
                        :].reset_index(drop=True)

        if mode!='none':
            ##########
            visit_dates = data_id[date_col].values
            relapse_df = pd.DataFrame([visit_dates]*len(relapse_id))
            relapse_df['relapse'] = relapse_id[rdate_col].values
            dist = relapse_df.drop(['relapse'],axis=1).subtract(relapse_df['relapse'], axis=0).apply(lambda x : pd.to_timedelta(x).dt.days)
            distm = - dist.mask(dist>0, other=float('nan'))
            distp = dist.mask(dist<0, other=float('nan'))
            relapse_id['closest_vis-'] = None if all(distm.isna()) else distm.idxmin(axis=1)
            relapse_id['closest_vis+'] = None if all(distp.isna()) else distp.idxmin(axis=1)
            ##########

            delta_raw, raw_dates = [], []
            last_conf = None

            for irel in range(len(relapse_id)):

                if last_conf is not None and last_conf>=relapse_id.loc[irel,rdate_col].date():
                    continue
                if verbose==2:
                    print('Relapse #%d / %d' %(irel+1, len(relapse_id)))
                change_idx = relapse_id.loc[irel,'closest_vis+']
                bl_idx = change_idx-1 if relapse_id.loc[irel,'closest_vis-']==change_idx else int(relapse_id.loc[irel,'closest_vis-'])

                if (np.isnan(change_idx)
                    or (data_id.loc[change_idx,date_col]
                        - relapse_id.loc[irel,rdate_col]).days > rel_infl # out of relapse influence
                    or data_id.loc[change_idx,value_col] - data_id.loc[bl_idx,value_col]
                        < delta(data_id.loc[bl_idx,value_col]) # no increase
                    ):
                    if verbose == 2:
                            print('No relapse-associated change')
                    continue
                else:
                    change_idx = int(change_idx)
                    conf_idx = next((x for x in range(change_idx+1, nvisits)
                                if conf_window[0] <= (data_id.loc[x,date_col] - data_id.loc[change_idx,date_col]).days
                                                             <= conf_window[1] # date in confirmation range
                                and data_id.loc[x,'closest_rel-'] > rel_infl), # out of relapse influence
                                None)

                    # CONFIRMED PROGRESSION:
                    # ---------------------
                    if (conf_idx is not None # confirmation visits available
                            and all([data_id.loc[x,value_col] - data_id.loc[bl_idx,value_col] >= delta(data_id.loc[bl_idx,value_col])
                                     for x in range(change_idx+1,conf_idx+1)]) # increase is confirmed at first valid date
                            ):
                        valid_prog = 1
                        if require_sust_months:
                            next_nonsust = next((x for x in range(conf_idx+1,nvisits) # next value found
                            if data_id.loc[x,value_col] - data_id.loc[bl_idx,value_col]
                                        < delta(data_id.loc[bl_idx,value_col]) # increase not sustained
                                            ), None)
                            valid_prog = (next_nonsust is None) or (data_id.loc[next_nonsust,date_col]
                                                        - data_id.loc[conf_idx,date_col]).days > require_sust_months*30.44
                        if valid_prog:
                            sust_idx = next((x for x in range(conf_idx+1,nvisits) # next value found
                                        if (data_id.loc[x,date_col] - data_id.loc[conf_idx,date_col]).days
                                        > require_sust_months*30.44
                                            ), None)
                            sust_idx = nvisits-1 if sust_idx is None else sust_idx-1 #conf_idx #
                            value_change = data_id.loc[change_idx:sust_idx+1,value_col].min() - data_id.loc[bl_idx,value_col]
                            # value_change = data_id.loc[change_idx,value_col] - data_id.loc[bl_idx,value_col]

                            delta_raw.append(value_change)
                            raw_dates.append(data_id.loc[change_idx,date_col].date())
                            if return_raw_dates:
                                raw_events.append([subjid, data_id.loc[change_idx,date_col].date()])
                            last_conf = data_id.loc[conf_idx,date_col].date()

                            if verbose == 2:
                                print('Relapse-associated confirmed progression on %s'
                                      %(data_id.loc[change_idx,date_col].date()))
                        else:
                            if verbose == 2:
                                print('Change confirmed but not sustained for >=%d months: proceed with search'
                                      %require_sust_months)

                    # NO confirmation:
                    # ----------------
                    else:
                        if verbose == 2:
                            print('Change not confirmed: proceed with search')

            if verbose == 2:
                print('Examined all relapses: end process')

            for d_value, date in zip(delta_raw, raw_dates):
                data_id.loc[data_id[date_col].apply(lambda x : x.date())>=date, ra_col]\
                    = data_id.loc[data_id[date_col].apply(lambda x : x.date())>=date, ra_col] + d_value

        if mode in ('ri', 'both'):
            data_id[ri_col] = data_id[value_col] - data_id[ra_col]
            if subtract_bl and len(data_id)>0:
                data_id[ri_col+'-bl'] = data_id[ri_col] - data_id.loc[0,ri_col]
        if subtract_bl and len(data_id)>0:
                data_id[value_col+'-bl'] = data_id[value_col] - data_id.loc[0,value_col]

        if return_rel_num and len(data_id)>0:
            for date in relapse_dates:
                data_id.loc[data_id[date_col].apply(lambda x : x.date())>=pd.to_datetime(date).date(),'relapse_num']\
                    = data_id.loc[data_id[date_col].apply(lambda x : x.date())>=pd.to_datetime(date).date(),'relapse_num'] + 1

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

    return (data_sep, pd.DataFrame(raw_events, columns=[subj_col,date_col])) if return_raw_dates else data_sep


#####################################################################################

def confirmed_value(data, value_col, date_col, idx=0, min_confirmed=None,
                   relapse=None, rdate_col=None, rel_infl=30,
                   conf_months=6, conf_tol_days=45, conf_left=False,
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
        rel_infl, int: influence of last relapse (days)
        conf_months, int: period before confirmation (months)
        conf_tol_days, int or list-like of length 1 or 2: tolerance window for confirmation visit (days): [t(months)-conf_tol[0](days), t(months)+conf_tol[0](days)]
        conf_left, bool: if True, confirmation window is [t(months)-conf_tol(days), inf)
        pira, bool: only confirm value if there are no relapses between value and confirmation
        verbose, int: 0[default, print no info], 1[print concise info], 2[print extended info]
    RETURNS:
        True if value is confirmed, False otherwise.
    """

    udates, ucounts = np.unique(data[date_col].values, return_counts=True)
    if any(ucounts>1):
        data = data.groupby(date_col).last()

    data = data.reset_index(drop=True)
    nvisits = len(data)

    if relapse is not None and rdate_col is None:
        rdate_col = date_col

    if relapse is not None:
        nrel = len(relapse)
        relapse_dates = relapse[rdate_col].values
        relapse_df = pd.DataFrame([relapse_dates]*len(data))
        relapse_df['visit'] = data[date_col].values
        dist = relapse_df.drop(['visit'],axis=1).subtract(relapse_df['visit'], axis=0).apply(lambda x : pd.to_timedelta(x).dt.days)
        distm = - dist.mask(dist>0, other= - float('inf'))
        distp = dist.mask(dist<0, other=float('inf'))
        data.insert(0, 'closest_rel-', float('inf') if all(distm.isna()) else distm.min(axis=1))
        data.insert(0, 'closest_rel+', float('inf') if all(distp.isna()) else distp.min(axis=1))

        all_dates, ii = np.unique(list(data[date_col].values) + list(relapse_dates),
                              return_index=True) # numpy unique() returns sorted values
        sorted_ind = np.arange(nvisits+nrel)[ii]
        is_rel = [x in relapse_dates for x in all_dates] # whether a date is a relapse
        # If there is a relapse with no visit, readjust the indices:
        date_dict = {sorted_ind[i] : i for i in range(len(sorted_ind))}
    else:
        data.insert(0, 'closest_rel-', float('inf'))
        data.insert(0, 'closest_rel+', float('inf'))
        is_rel = [False for x in data[date_col].values]
        date_dict = {i : i for i in range(len(is_rel))}

    if milestone is None:
        milestone = data.loc[idx,value_col]

    conf_window = (int(conf_months*30.5) - conf_tol_days, float('inf') if conf_left
                   else int(conf_months*30.5) + conf_tol_days)

    conf_idx = next((x for x in range(idx+1, nvisits)
            if conf_window[0] <= (data.loc[x,date_col] - data.loc[idx,date_col]).days <= conf_window[1] # date in confirmation range
            and data.loc[x,'closest_rel-'] > rel_infl), # out of relapse influence
            None)
    if conf_idx is not None and all([data.loc[x,value_col]
                >= min(milestone, data.loc[idx,value_col]) for x in range(idx+1,conf_idx+1)]):
        valid = not any(is_rel[date_dict[idx]:date_dict[conf_idx]+1]) if pira else True
        return valid
    else:
        return False
