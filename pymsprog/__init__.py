
__version__ = "1.0.0"

import numpy as np
import pandas as pd

import datetime

import copy

from importlib import resources

import warnings

from importlib import resources

#####################################################################################
def load_toy_data():
    '''
    Load the example longitudinal MS dataset provided with ``pymsprog``.

    This toy dataset is intended for testing and demonstration purposes.
    It provides artificially generated Extended Disability Status Scale (EDSS) and
    Symbol Digit Modalities Test (SDMT) longidutinal scores, visit dates, and relapse onset dates
    in a small cohort of example patients to illustrate the use of the package.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        A ``visits`` sheet containing disability info, and a ``relapses`` sheet containing
        relapse info.

    Notes
    -----
    The dataset is loaded from an Excel file (``MSprog_toydata.xlsx``) bundled with the package.
    The ``visits`` sheet includes the following columns.

    - ``id``: subject identifier.
    - ``date``: visit date.
    - ``EDSS``: EDSS score.
    - ``SDMT``: SDMT score.

    The ``relapses`` sheet includes the following columns.

    - ``id``: subject identifier.
    - ``date``: relapse onset date.

    Examples
    --------
    >>> visits, relapses = load_toy_data()
    >>> visits.head()
    >>> relapses.head()
    '''
    with resources.path('pymsprog.data', 'MSprog_toydata.xlsx') as data_path:
        return pd.read_excel(data_path, sheet_name='visits'), pd.read_excel(data_path, sheet_name='relapses')


#####################################################################################

def MSprog(data, subj_col, value_col, date_col, outcome,
            relapse=None, rsubj_col=None, rdate_col=None,  # renddate_col=None,
            subjects=None, delta_fun=None, worsening=None, valuedelta_col=None,
            event='firstCDW', baseline='fixed', proceed_from='firstconf',
            sub_threshold_rebl='none',
            bl_geq=False, relapse_rebl=False,  skip_local_extrema='none',
            validconf_col=None, conf_days=12 * 7, conf_tol_days=[7, 2 * 365.25],
            require_sust_days=0, check_intermediate=True,
            relapse_to_bl=30, relapse_to_event=0, relapse_to_conf=30,
            relapse_assoc=90, relapse_indep=None,
            impute_last_visit=0,
            include_dates=False, include_value=False, include_stable=True, verbose=1):
    '''

    Assess multiple sclerosis disability course from longitudinal data.

    The function detects and characterises the confirmed disability worsening (CDW)
    or improvement events of an outcome measure (EDSS, NHPT, T25FW, or SDMT; or any custom outcome)
    for one or more subjects, based on repeated assessments
    through time (and on the dates of acute episodes, if any).
    Several qualitative and quantitative options are given as arguments that can be set
    by the user and reported as a complement to the results to ensure reproducibility.

    Parameters
    ----------
    data: pandas.DataFrame
        Longitudinal data containing subject ID, outcome value, date of visit.

    subj_col: str
        Name of data column with subject ID.

    value_col: str
        Name of data column with outcome value.

    date_col: str
        Name of data column with date of visit.

    outcome: str
        Outcome type. Must be one of the following.

        - 'edss' (Expanded Disability Status Scale)
        - 'nhpt' (Nine-Hole Peg Test)
        - 't25fw' (Timed 25-Foot Walk)
        - 'sdmt' (Symbol Digit Modalities Test)
        - None (only accepted when specifying a custom ``delta_fun``).

    relapse: pandas.DataFrame
        Longitudinal data containing subject ID and relapse onset dates (if any).

    rsubj_col: str
        Name of subject ID column for relapse data, if different from outcome data.

    rdate_col: str
        Name of onset date column for relapse data, if different from outcome data.

    subjects: array-like
        Subset of subject IDs to analyse. If none is specified, all subjects listed in ``data`` are included.

    delta_fun: function
        Custom function specifying the minimum shift in the outcome measure that counts as a valid change
        from the provided reference value.
        The function provided must take a numeric value (reference score) as input, and return a numeric
        value corresponding to the minimum shift from baseline.
        If none is specified (default), the user must provide a non-None value for the ``outcome`` argument
        in order to use the built-in function :func:`compute_delta()`.

    worsening: str
        The direction of worsening ('increase' if higher values correspond to worse disease course, 'decrease' otherwise).
        This argument is only used when ``outcome`` is set to None. Otherwise, ``worsening`` is automatically set to
        'increase' if ``outcome`` is set to 'edss', 'nhpt', 't25fw', and to 'decrease' if ``outcome`` is set to 'sdmt'.

    valuedelta_col: str
        Name of data column with alternative outcome value to use as baseline for delta function.
        It may be used, for instance, when analysing *partial* disability trajectories extracted
        using :func:`separate_ri_ra()`.
        Example:\n
        Original trajectory: EDSS = 3, 3.5, 5.5, 5, 5, 5.5, 6\n
        Relapse-independent trajectory: RIEDSS = 3, 3.5, 3.5, 3.5, 3.5, 4, 4.5 (obtained using :func:`separate_ri_ra()`)\n
        To extract progression events from the relapse-independent trajectory, the delta function should be referred
        to the original EDSS values (e.g., the worsening of 0.5 from 4 to 4.5 should be considered valid because
        the actual disability score of the patient went from 5.5 to 6).
        In such a case, :func:`MSprog()` should be used with ``value_col='RIEDSS'`` and ``valuedelta_col='EDSS'``.

    event: str
        Specifies which events to detect. Must be one of the following.

        - 'firstCDW' (first confirmed disability worsening (CDW), default);
        - 'first' (only the very first confirmed event -- improvement or worsening);
        - 'firsteach' (first confirmed disability improvement and first CDW -- in chronological order);
        - 'firstCDWtype' (first CDW of each kind -- PIRA, RAW, and undefined, in chronological order);
        - 'firstPIRA' (first PIRA);
        - 'firstRAW' (first RAW);
        - 'multiple' (all events in chronological order).

    baseline: str
        Specifies the baseline scheme. Must be one of the following.

        - 'fixed': first valid outcome value, default;
        - 'roving_impr': updated after every confirmed disability improvement (to the visit determined by ``proceed_from``);
          suitable for a first-CDW setting to discard fluctuations around baseline -- not recommended for randomised data;
        - 'roving_wors': updated after every CDW (to the visit determined by ``proceed_from``);
          suitable when searching for a specific type of CDW (i.e., when ``event`` is set to 'firstPIRA' or 'firstRAW');
        - 'roving': updated after each improvement or worsening event to the visit determined by ``proceed_from``;
          suitable for a multiple-event setting (i.e., when ``event`` is set to 'multiple', 'firsteach', or 'firstCDWtype')
          or when searching for a specific type of CDW (i.e., when ``event`` is set to 'firstPIRA' or 'firstRAW')
          -- not recommended for randomised data.

    proceed_from: str
        After detecting a confirmed disability event, continue searching:

        - from the next visit after the first qualifying confirmation visit if ``proceed_from='firstconf'``;
        - from the next visit after the event if ``proceed_from='event'``.

        If ``baseline`` is set to 'roving', 'roving_impr', or 'roving_wors', when rebaselining after a
        confirmed disability event, the baseline is moved to:

        - the first qualifying confirmation visit if ``proceed_from='firstconf'``;
        - the event visit if ``proceed_from='event'``.

    sub_threshold_rebl: str
        This argument is only used if ``baseline`` is not set to 'fixed'.
        Must be one of the following.

        - 'event': any confirmed sub-threshold event (i.e., any *confirmed* change in the outcome measure,
          regardless of ``delta_fun``) can potentially trigger a re-baseline;
        - 'improvement': any confirmed sub-threshold improvement (i.e., any *confirmed* improvement
          in the outcome measure, regardless of ``delta_fun``) can potentially trigger a re-baseline;
        - 'worsening': any confirmed sub-threshold worsening (i.e., any *confirmed* worsening in the
          outcome measure, regardless of ``delta_fun``) can potentially trigger a re-baseline;
        - 'none': only use valid confirmed events (as per ``delta_fun``) for rebaseline.

    bl_geq: bool
        This argument is only used if the baseline is moved.
        If True, the new reference value must always be greater or equal than the previous one;
        when it is not, the old reference value is assigned to it [Kappos JAMA Neurol 2020].

    relapse_rebl: bool
        If True, re-baseline after every relapse.

    skip_local_extrema: str
        This argument is only used when moving the baseline. It controls re-baseline behaviour
        in the presence of local minima or maxima of the outcome trajectory.
        A visit ``i`` is a local minimum point for ``outcome`` if ``outcome[i+1]>outcome[i]`` and ``outcome[i-1]>outcome[i]``;
        local maxima are defined similarly.\n
        A visit ``i`` is a *strict* local minimum point for ``outcome`` if:\n
        ``outcome[i+1]-outcome[i]>=delta_fun(outcome[i])``;\n
        ``outcome[i-1]-outcome[i]>=delta_fun(outcome[i])``.\n
        Strict local maxima are defined similarly.\n
        When ``outcome[i]=outcome[i-2]``, visit ``i`` is *not* considered a local extremum point even if the above conditions hold.
        This controls for cases where the outcome has an undulating course.\n
        The following argument values are accepted.
        
        - 'none': local extrema are always accepted as valid baseline values;
        - 'delta': the baseline cannot be placed at a *strict* local minimum or maximum;
        - 'all': the baseline cannot be placed at a local minimum or maximum.

    validconf_col: str
        Name of data column specifying which visits can (True) or cannot (False) be used as confirmation visits.
        The input data does not necessarily have to include such a column.
        If not specified (``validconf_col=None``), all visits are potentially used as confirmation visits.

    conf_days: int, float, or array-like
        Period before confirmation (days). Can be a single value or array-like of any length if multiple
        windows are to be considered.

    conf_tol_days: int, float, or array-like of length 1 or 2
        Tolerance window for confirmation visit (days); can be a single value (same tolerance on left and right)
        or array-like of length 2 (different tolerance on left and right).
        The right end of the interval can be set to ``float('inf')`` (confirmation window unbounded on the right
        -- e.g., "confirmed over 12 *or more* weeks").

    require_sust_days: int or float
        Minimum number of days over which a confirmed change must be sustained
        (i.e., confirmed at *all* visits occurring in the specified period) to be retained as an event.
        Events sustained for the remainder of the follow-up period are always retained regardless of follow-up duration.
        Setting ``require_sust_days=float('inf')``, events are retained only when sustained for the remainder of the follow-up period.
        (Warning: if ``check_intermediate`` is set to False, sustained change will be established based
        *only on the end* of the specified period.)

    check_intermediate: bool
        If True (default), events are confirmed *over all intermediate visits* up to the confirmation visit.
        If set to False (not recommended in most cases, as it may discard meaningful fluctuations),
        events will be confirmed *only at* the specified confirmation visit
        (and *only at the end* of the period defined by ``require_sust_days``, if any).

    relapse_to_bl: int, float, or array-like of length 1 or 2
        Minimum distance from a relapse (days) for a visit to be used as baseline.
        Can be a single value (minimum distance from *last* relapse) or array-like of length 2
        (minimum distance from *last* relapse, minimum distance from *next* relapse).
        Note that setting the distance to zero means keeping the baseline where it is regardless of surrounding relapses.
        # If relapse end dates are available (``renddate_col``), the minimum distance from last relapse
        # is overwritten by the relapse duration, unless it was set to zero (in which case it stays 0).
        If the designated baseline does not respect this constraint, the baseline is moved to the next available visit.

    relapse_to_event: int, float, or array-like of length 1 or 2
        Minimum distance from a relapse (days) for an event to be considered as such.
        Can be a single value (minimum distance from *last* relapse) or array-like of length 2
        (minimum distance from *last* relapse, minimum distance from *next* relapse).
        Note that setting the distance to zero means retaining the event regardless of surrounding relapses.
        # If relapse end dates are available (``renddate_col``), the minimum distance from last relapse
        # is overwritten by the relapse duration, unless it was set to zero (in which case it stays 0).

    relapse_to_conf: int, float, or array-like of length 1 or 2
        Minimum distance from a relapse (days) for a visit to be a valid confirmation visit.
        Can be a single value (minimum distance from *last* relapse) or array-like of length 2
        (minimum distance from *last* relapse, minimum distance from *next* relapse).
        Note that setting the distance to zero means using any visit for confirmation regardless of surrounding relapses.
        # If relapse end dates are available (``renddate_col``), the minimum distance from last relapse
        # is overwritten by the relapse duration, unless it was set to zero (in which case it stays 0).

    relapse_assoc: int, float, or array-like of length 1 or 2
        Maximum distance from a relapse (days) for a CDW event to be classified as RAW.
        Can be a single value (maximum distance from *last* relapse) or array-like of length 2
        (maximum distance from *last* relapse, maximum distance from *next* relapse).
        # If relapse end dates are available (``renddate_col``), the maximum distance from last relapse
        # is overwritten by the relapse duration.

    relapse_indep: dict
        Specifies relapse-free intervals for PIRA definition. Must be given in the form:
        ``{'prec': (p0, p1), 'event': (e0, e1), 'conf': (c0, c1)}``.
        The dictionary specifies the intervals around (any subset of) the three checkpoints:

        - 'prec': a visit preceding the event -- see below;
        - 'event': the disability worsening event;
        - 'conf': the first available confirmation visit.

        The dictionary can also optionally contain a key-value pair specifying how to choose 'prec':

        - ``'prec_type': 'baseline'`` → 'prec' is the current baseline;
        - ``'prec_type': 'last'`` → 'prec' is the last visit before the event;
        - ``'prec_type': 'last_lower'`` → 'prec' is the last pre-worsening visit:
          ``i`` such that ``outcome[event] - outcome[i] >= delta_fun(outcome[i])`` (if ``worsening='increase'``, the opposite otherwise)
          and same for the confirmation visit.
        
        If 'prec_type' is not in the dictionary keys, 'prec' is assumed to be the current baseline.

        If both ends of an interval are 0 (e.g., if both ``p0=0`` and ``p1=0``), the checkpoint is ignored.
        If the right end is None, the interval is assumed to extend up to the left end of the next interval.
        If the left end is None, the interval is assumed to extend up to the right end of the previous interval.
        # If relapse end dates are available (``renddate_col``), it is possible to also define PIRA based on those...

    impute_last_visit: float or int
        Imputation probability for worsening events occurring at last visit (i.e. with no confirmation).
        Unconfirmed worsening events occurring at the last visit are never imputed if ``impute_last_visit=0``;
        they are always imputed if ``impute_last_visit=1``;
        they are imputed with probability ``p``, ``0<p<1``, if ``impute_last_visit=p``.
        If a value ``N>1`` is passed, unconfirmed worsening events are imputed only if occurring within
        ``N`` days of follow-up (e.g., in case of early discontinuation).

    include_dates: bool
        If True, the ``results`` DataFrame will include the date of each event ('date' column)
        and the date of the corresponding baseline ('bldate' column).

    include_value: bool
        If True, the ``results`` DataFrame will include the outcome value at each event ('value' column)
        and at the corresponding baseline ('blvalue' column).

    include_stable: bool
        If True, subjects with no confirmed events are included in the ``results`` DataFrame,
        with 'time2event' = total follow up.

    verbose: int
        One of: 0 (print no info); 1 (print concise info, default); 2 (print extended info).

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
         - ``summary``: summary of detected events for each subject;
         - ``results``: extended info on event sequence for each subject.

    Notes
    -----
    The events are detected sequentially by scanning the outcome values in chronological order.
    Valid time windows for confirmation visits are determined by arguments
    ``conf_days``, ``conf_tol_days``, ``relapse_to_conf``.
    CDW events are classified as relapse-associated or relapse-independent based on their relative timing
    with respect to the relapses. Specifically, relapse-associated worsening (RAW) events are defined as
    CDW events occurring within a specified interval (``relapse_assoc`` argument) from a relapse;
    the definition of progression independent of relapse activity (PIRA) is established
    by specifying relapse-free intervals (``relapse_indep`` argument).

    Raises
    ------
    ValueError
        If the arguments are incorrectly specified.

    Examples
    --------
    >>> toydata_visits, toydata_relapses = load_toy_data()
    >>> # 1. EDSS course
    >>> summary, results = MSprog(toydata_visits, relapse=toydata_relapses, subj_col='id', value_col='EDSS', date_col='date', outcome='edss')
    >>> # 1. SDMT course
    >>> summary, results = MSprog(toydata_visits, relapse=toydata_relapses, subj_col='id', value_col='EDSS', date_col='date', outcome='sdmt')

    '''

    #####################################################################################
    # SETUP

    #### for debugging
    #np.seterr(all='raise')
    ####

    warning_msgs = []

    ###########################
    # CHECKS ON ARGUMENT VALUES

    # If conf_days is a single value, make it a list with a single element
    try:
       _ = (e for e in conf_days) # check if conf_days is iterable
    except TypeError:
       conf_days = [conf_days]

    # If conf_tol_days is a single value, duplicate it (equal left and right tolerance)
    try:
        _ = (e for e in conf_tol_days) # check if it's iterable
        if len(conf_tol_days) == 1:
            conf_tol_days = [conf_tol_days[0], conf_tol_days[0]]
    except TypeError:
        conf_tol_days = [conf_tol_days, conf_tol_days]

    # If relapse_to_bl is a single value, set right bound to zero
    try:
        _ = (e for e in relapse_to_bl)  # check if it's iterable
        if len(relapse_to_bl) == 1:
            relapse_to_bl = [relapse_to_bl[0], 0]
    except TypeError:
        relapse_to_bl = [relapse_to_bl, 0]

    # If relapse_to_event is a single value, set right bound to zero
    try:
        _ = (e for e in relapse_to_event)  # check if it's iterable
        if len(relapse_to_event) == 1:
            relapse_to_event = [relapse_to_event[0], 0]
    except TypeError:
        relapse_to_event = [relapse_to_event, 0]

    # If relapse_to_conf is a single value, set right bound to zero
    try:
        _ = (e for e in relapse_to_conf)  # check if it's iterable
        if len(relapse_to_conf) == 1:
            relapse_to_conf = [relapse_to_conf[0], 0]
    except TypeError:
        relapse_to_conf = [relapse_to_conf, 0]

    # If relapse_assoc is a single value, set right bound to zero
    try:
        _ = (e for e in relapse_assoc)  # check if it's iterable
        if len(relapse_assoc) == 1:
            relapse_assoc = [relapse_assoc[0], 0]
    except TypeError:
        relapse_assoc = [relapse_assoc, 0]

    if outcome is None or outcome.lower() not in ['edss', 'nhpt', 't25fw', 'sdmt']:
        outcome = 'outcome'
    else:
        outcome = outcome.lower()

    if event not in ('firstCDW', 'first', 'firsteach', 'firstCDWtype',
    'firstPIRA', 'firstRAW', 'multiple'):
        raise ValueError('Invalid value for `event` argument. Valid values: \'firstCDW\', \'first\', '
                            + '\'firsteach\', \'firstCDWtype\', \'firstPIRA\', \'multiple\'.')

    if baseline not in ('fixed', 'roving_impr', 'roving_wors', 'roving'):
        raise ValueError('Invalid value for `baseline` argument. Valid values: '
                        + '\'fixed\', \'roving_impr\', \'roving_wors\', \'roving\'.')

    if proceed_from not in ('event', 'firstconf'):
        raise ValueError('Invalid value for `proceed_from` argument. Valid values: \'event\', \'firstconf\'.')

    if sub_threshold_rebl not in ('event', 'improvement', 'worsening', 'none'):
        raise ValueError('Invalid value for `sub_threshold_rebl` argument. Valid values: '
                        + '\'event\', \'improvement\', \'worsening\', \'none\'.')

    if skip_local_extrema not in ('none', 'delta', 'all'):
        raise ValueError('Invalid value for `skip_local_extrema` argument. Valid values: \'none\', \'delta\', \'all\'.')

    if relapse_indep is None:
        relapse_indep = {'prec': (0, 0), 'event': (90, 30), 'conf': (90, 30), 'prec_type': 'baseline'}
    if (not isinstance(relapse_indep, dict) or 'prec' not in relapse_indep.keys() or
            'event' not in relapse_indep.keys() or 'conf' not in relapse_indep.keys()):
        raise ValueError('Invalid value for `relapse_indep` argument. '
                + 'Must be a dictionary containing at least the keys \'prec\', \'event\', \'conf\'.')
    if 'prec_type' not in relapse_indep.keys():
        relapse_indep['prec_type'] = 'baseline'
    if relapse_indep['prec_type'] not in ('baseline', 'last', 'last_lower'):
        raise ValueError('Invalid value for `relapse_indep[\'prec_type\']`. Valid values: \'baseline\', \'last\', \'last_lower\'.')

    # end of checks
    ###########################

    data = data.copy()

    # If no column names are specified for the relapse file, use the main ones
    if rsubj_col is None:
        rsubj_col = subj_col
    if rdate_col is None:
        rdate_col = date_col

    # If no `validconf_col` is specified, create a dummy one
    if validconf_col is None:
        validconf_col = 'validconf'
        data.loc[:, validconf_col] = 1

    # If no `valuedelta_col` is specified, copy `value_col`
    if valuedelta_col is None:
        valuedelta_col = value_col

    if impute_last_visit < 0:
        raise ValueError('`impute_last_visit` must be nonnegative')
    elif impute_last_visit <= 1:
        # If impute_last_visit is a probability, set no limit to follow-up length (Inf)
        impute_max_fu = float('inf')
    else:
        # If impute_last_visit is a follow-up time, save the value and set probability to 1
        impute_max_fu = impute_last_visit
        impute_last_visit = 1

    if outcome in ('edss', 'nhpt', 't25fw'):
        worsening = 'increase'
    elif outcome=='sdmt':
        worsening = 'decrease'
    elif worsening is None:
        raise ValueError('Either specify an outcome type, or specify the direction of worsening (\'increase\' or \'decrease\')')

    def isevent_loc(x, bl, type='wors', st=False, baseline_delta=None):
        return is_event(x, bl, type=type, outcome=outcome, worsening=worsening,
                        sub_threshold=st, delta_fun=delta_fun, baseline_delta=baseline_delta)

    # Remove missing values from columns of interest
    data = data[np.unique([subj_col, value_col, date_col, validconf_col, valuedelta_col])].copy().dropna()  #_c_#

    # Convert dates to datetime.date format
    data[date_col] = pd.to_datetime(data[date_col])
    if relapse is None:
        relapsedata = False
        relapse_rebl = False
        relapse = pd.DataFrame([], columns=[rsubj_col, rdate_col])
        relapse_start = data[date_col].min()
    else:
        relapsedata = True
        relapse = relapse[[rsubj_col, rdate_col]].copy().dropna()
        relapse[rdate_col] = pd.to_datetime(relapse[rdate_col])
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
    # Assess disability course

    all_subj = data[subj_col].unique()
    nsub = len(all_subj)
    max_nevents = round(data.groupby(subj_col)[date_col].count().max()/2)
    results = pd.DataFrame([[None]*10 + [None]*len(conf_days)
                            + [None]*(len(conf_days)) + [None]*2]*nsub*max_nevents,
               columns=[subj_col, 'nevent', 'event_type', 'bldate', 'blvalue', 'date', 'value',
                        'total_fu', 'time2event', 'bl2event']
                       + ['conf'+str(m) for m in conf_days]+ ['PIRA_conf'+str(m) for m in conf_days]
                       + ['sust_days', 'sust_last'])
    results[subj_col] = np.repeat(all_subj, max_nevents)
    results['nevent'] = np.tile(np.arange(1, max_nevents + 1), nsub)
    summary = pd.DataFrame([['']+[0]*5]*nsub, columns=['event_sequence', 'improvement', 'CDW',
                                                  'RAW', 'PIRA', 'undefined_CDW'], index=all_subj)
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
            print('Values not in chronological order: sorting them automatically')  #raise TypeError('uffa')
            # print(sorted_tmp)
            # print(data_id)
            data_id = sorted_tmp.copy()

        nvisits = len(data_id)
        first_visit = data_id[date_col].min()
        relapse_id = relapse.loc[relapse[rsubj_col]==subjid,:].copy().reset_index(drop=True)
        relapse_id = relapse_id.loc[relapse_id[rdate_col] >= first_visit - relapse_to_bl[0], :]  # ignore relapses occurring before first visit
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
        conf, sustd, sustl = {m : [] for m in conf_days}, [], []
        pira_conf = {m : [] for m in conf_days} #[1:]}  #_piraconf_#


        bl_idx, search_idx = 0, 1 # baseline index and index of where we are in the search
        proceed = 1
        conf_window = [(int(c) - conf_tol_days[0], int(c) + conf_tol_days[1]) for c in conf_days]
        irel = 0 if nrel==0 else next((r for r in range(nrel) if relapse_dates[r] > data_id.loc[bl_idx, date_col]), None)
        bl_last = None

        if verbose == 2:
            print(f'Baseline at visit no.{bl_idx + 1}')

        while proceed:

            # Set baseline (skip if local extremum or within relapse influence)
            local_extr = True  # to enter the loop
            while proceed and (data_id.loc[bl_idx, 'closest_rel-'] < relapse_to_bl[0]
                    or data_id.loc[bl_idx, 'closest_rel+'] < relapse_to_bl[1]
                    or local_extr):

                # Check if baseline is local extremum:
                if skip_local_extrema != 'none':
                    prec = data_id.loc[bl_idx, :] if bl_idx == 0 else data_id.loc[bl_idx - 1, :]
                    prec2value = -1 if bl_idx <= 1 else data_id.loc[bl_idx - 2, value_col]
                    subs = data_id.loc[bl_idx, :] if bl_idx == nvisits - 1 else data_id.loc[bl_idx + 1, :]
                    vis = data_id.loc[bl_idx, :]
                    local_extr = ((
                              isevent_loc(prec[value_col], vis[value_col], type='wors',
                                          st=skip_local_extrema == 'all',
                                          baseline_delta=vis[valuedelta_col])
                              and isevent_loc(subs[value_col], vis[value_col], type='wors',
                                              st=skip_local_extrema == 'all',
                                              baseline_delta=vis[valuedelta_col])
                                  ) or (
                              isevent_loc(prec[value_col], vis[value_col], type='impr',
                                          st=skip_local_extrema == 'all',
                                          baseline_delta=vis[valuedelta_col])
                              and isevent_loc(subs[value_col], vis[value_col], type='impr',
                                              st=skip_local_extrema == 'all',
                                              baseline_delta=vis[valuedelta_col])
                                  )) and (vis[value_col] != prec2value)
                else:
                    local_extr = False

                if (data_id.loc[bl_idx,'closest_rel-'] >= relapse_to_bl[0]
                    and data_id.loc[bl_idx,'closest_rel+'] >= relapse_to_bl[1]
                    and not local_extr):
                    # If baseline is out of relapse influence and not a local extremum, keep it
                    break

                if verbose==2:
                    print(f'Baseline (visit no.{bl_idx + 1}) is {"a local extremum" if local_extr else "within relapse influence"}: '
                            + f'moved to visit no.{bl_idx + 2}')
                bl_idx += 1
                if bl_idx > nvisits - 2:
                    proceed = 0
                    if verbose == 2:
                        print('Not enough visits left: end process')

            # If `relapse_rebl` is enabled, update relapse index to next relapse after baseline
            if relapse_rebl and irel is not None and irel < nrel and bl_idx < nvisits:
                irel = next((x for x in range(irel, nrel)  # next relapse
                             if relapse_dates[x] > data_id.loc[bl_idx, date_col]  # after current baseline
                             ), None)

            # If baseline was moved after current search index, move search index:
            search_idx = bl_idx + 1 if search_idx <= bl_idx else search_idx
            if verbose == 2:
                print(f'Searching for events from visit no.{search_idx + 1 if search_idx < nvisits else "-"} on')


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
                    if isevent_loc(data_id.loc[x, value_col], bl[value_col], type='change',
                        st=sub_threshold_rebl!='none', baseline_delta=data_id.loc[bl_idx, valuedelta_col]) # first occurring value!=baseline
                        and (data_id.loc[x, 'closest_rel-'] >= relapse_to_event[0])  # occurring out of influence of last relapse
                        and (data_id.loc[x, 'closest_rel+'] >= relapse_to_event[1])  # occurring out of influence of next relapse
                               ), None)
            if change_idx is None: # value does not change in any subsequent visit
                conf_idx = []
                conf_t = {}
                proceed = 0
                if verbose == 2:
                    print('No %s change in any subsequent visit: end process' %outcome.upper())
            elif (relapse_rebl and irel is not None
                  and irel < nrel
                  and data_id.loc[change_idx, date_col]
                        > relapse_dates[irel] + (0 if event=='firstPIRA' else relapse_assoc[0])
                ):
                # If `relapse_rebl`is enabled and the detected change from baseline has crossed the next relapse,
                # the baseline needs to be moved after that relapse (post-relapse re-baseline below).
                # (unless search_idx is after the relapse but within its influence as per `relapse_assoc[0]`: could be a RAW)
                search_idx = change_idx
            else:
                conf_idx = [[x for x in range(change_idx + 1, nvisits)
                        if c[0] <= data_id.loc[x, date_col] - data_id.loc[change_idx,date_col] <= c[1]  # date in confirmation range
                        and data_id.loc[x,'closest_rel-'] >= relapse_to_conf[0]  # occurring out of influence of last relapse
                        and data_id.loc[x,'closest_rel+'] >= relapse_to_conf[1]  # occurring out of influence of next relapse
                        and data_id.loc[x, validconf_col]]  # can be used as confirmation
                        for c in conf_window]
                conf_t = {conf_days[i] : conf_idx[i] for i in range(len(conf_days))}
                conf_idx = np.unique([x for i in range(len(conf_idx)) for x in conf_idx[i]])
                if verbose == 2:
                    print('%s change at visit no.%d (%s); potential confirmation visits available: no.%s'
                          %(outcome.upper(), change_idx + 1 ,
                            global_start + datetime.timedelta(days=data_id.loc[change_idx,date_col].item()), #_d_#
                            ', '.join(['%d' %(i + 1) for i in conf_idx])))

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
                    ):
                    next_change = next((x for x in range(conf_idx[0] + 1, nvisits)
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
                    if require_sust_days:
                        if not check_intermediate and ((data_id.loc[nvisits-1,date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_days):
                            sust_vis = next((x for x in range(change_idx+1, nvisits) if (data_id.loc[x,date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_days))
                        else:
                            sust_vis = nvisits - 1
                        valid_impr = ((next_nonsust is None) or (data_id.loc[next_nonsust,date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_days #.days #_d_# # improvement sustained up to end of follow-up, or for `require_sust_days` days
                                      ) if check_intermediate else isevent_loc(data_id.loc[sust_vis,value_col], # improvement confirmed at last visit, or first visit after `require_sust_days` days
                                    bl[value_col], type='impr', baseline_delta=data_id.loc[bl_idx,valuedelta_col])

                    # If the event is retained (as per `require_sust_days`), we store the info:
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
                        for m in conf_days:
                            confirmed_at = np.intersect1d(conf_t[m], conf_idx)
                            if len(confirmed_at)==0:
                                del conf_t[m]
                            conf[m].append(1 if len(confirmed_at)>0 else 0) #_conf_# 1 if m in conf_t else 0
                        for m in conf_days: #[1:]: #_piraconf_#
                            pira_conf[m].append(None)
                        sustd.append(data_id.loc[sust_idx,date_col] - data_id.loc[change_idx,date_col]) #.days #_d_#
                        sustl.append(int(sust_idx == nvisits-1)) #int(data_id.loc[nvisits-1,value_col] - bl[value_col] <= - delta(bl[value_col]))

                        # Print progress info
                        if verbose == 2:
                            print('%s improvement (visit no.%d, %s) confirmed at %s weeks, sustained up to visit no.%d (%s)'
                                  %(outcome.upper(), change_idx+1,
                                    global_start + datetime.timedelta(days=data_id.loc[change_idx,date_col].item()), #_d_#
                                    ', '.join([str(x) for x in conf_t.keys()]),  #_conf_#
                                    sust_idx+1,
                                    global_start + datetime.timedelta(days=data_id.loc[sust_idx,date_col].item()))) #_d_#

                    else:
                        # If the event is NOT retained (as per `require_sust_days`), we proceed.
                        if verbose == 2:
                            print('Change confirmed but not sustained for',
                                  ('>=%d days: ' %require_sust_days if require_sust_days<float('inf') else 'entire follow up:'),
                                    'proceed with search')

                    # For each m in conf_days, only keep the earliest available confirmation visit:
                    conf_idx = [np.min([x for x in conf_t[m] if x in conf_idx]) for m in
                                conf_t.keys()]  # _conf_#

                    if baseline in ('roving', 'roving_impr'):
                        # In a roving baseline setting, the baseline is moved after the confirmed event (even if it is not sustained):
                        newref = conf_idx[0] if proceed_from == 'firstconf' else change_idx
                        bl_idx = newref
                    elif valid_impr:
                        newref = conf_idx[0] if proceed_from == 'firstconf' else change_idx
                    else:
                        # If the event is not retained (as per `require_sust_days`),
                        # proceed with search starting from event, regardless of `proceed_from`:
                        newref = change_idx
                    if verbose == 2 and baseline != 'fixed':
                        print(f'Baseline at visit no.{bl_idx + 1}')

                    # Move the search index.
                    search_idx = newref + 1

                # Confirmed sub-threshold improvement: RE-BASELINE
                # ------------------------------------------------
                elif (len(conf_idx) > 0 # confirmation visits available
                        and data_id.loc[change_idx,value_col]<bl[value_col] # value decreased from baseline
                        and (all([data_id.loc[x,value_col]<bl[value_col]
                                 for x in range(change_idx+1,conf_idx[0]+1)])  # decrease is confirmed
                        if check_intermediate else data_id.loc[conf_idx[0],value_col]<bl[value_col])
                        and baseline in ('roving', 'roving_impr') and sub_threshold_rebl in ('event', 'improvement')
                        ):

                    newref = conf_idx[0] if proceed_from == 'firstconf' else change_idx

                    # Set new baseline after event:
                    bl_idx = newref
                    # Move search index
                    search_idx = newref + 1

                    if verbose == 2:
                        print('Confirmed sub-threshold %s improvement (visit no.%d)'
                              %(outcome.upper(), change_idx + 1))
                        print(f'Baseline at visit no.{bl_idx + 1}')

                # CONFIRMED WORSENING:
                # -------------------
                elif (isevent_loc(data_id.loc[change_idx, value_col], bl[value_col], type='wors',
                                        baseline_delta=data_id.loc[bl_idx, valuedelta_col])  # value worsened (>delta) from baseline
                    and ((len(conf_idx) > 0 # confirmation visits available
                        and (all([isevent_loc(data_id.loc[x, value_col], bl[value_col], type='wors',
                                              baseline_delta=data_id.loc[bl_idx, valuedelta_col]) #_c_#
                                 for x in range(change_idx + 1, conf_idx[0] + 1)])  # worsening is confirmed at (all visits up to) first valid date
                            if check_intermediate else isevent_loc(data_id.loc[conf_idx[0], value_col], bl[value_col],
                                type='wors', baseline_delta=data_id.loc[bl_idx, valuedelta_col]))  # worsening is confirmed at first valid date
                        ) or (data_id.loc[change_idx, date_col] - data_id.loc[0, date_col] <= impute_max_fu
                              and np.random.binomial(1, impute_last_visit, 1)
                              and change_idx == nvisits - 1))
                      ):

                    if change_idx == nvisits - 1:  # i.e., when imputing event at last visit
                        conf_idx = [nvisits - 1]

                    # First visit at which worsening is not sustained:
                    next_nonsust = next((x for x in range(conf_idx[0] + 1, nvisits) #_r_# #conf_idx[-1]
                        if not isevent_loc(data_id.loc[x,value_col], bl[value_col], type='wors',
                                           baseline_delta=data_id.loc[bl_idx, valuedelta_col]) #_c_# # increase not sustained
                                    ), None)
                    # Discard potential confirmation visits if these occur out of sustained interval:
                    conf_idx = conf_idx if next_nonsust is None else [ic for ic in conf_idx if
                                                                     ic < next_nonsust]  # confirmed dates

                    # The confirmed worsening can still be rejected if `require_sust_days>0`.
                    # The `valid_prog` flag indicates whether the event can (1) or cannot (0) be retained:
                    valid_prog = 1
                    if require_sust_days:
                        if not check_intermediate and ((data_id.loc[nvisits - 1, date_col]
                                    - data_id.loc[change_idx, date_col]) >= require_sust_days):
                            sust_vis = next((x for x in range(change_idx + 1, nvisits) if (data_id.loc[x,date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_days))
                        else:
                            sust_vis = nvisits - 1
                        valid_prog = ((next_nonsust is None) or (data_id.loc[next_nonsust, date_col]
                                    - data_id.loc[change_idx,date_col]) >= require_sust_days #.days #_d_# # worsening sustained up to end of follow-up, or for `require_sust_days` days
                                      ) if check_intermediate else isevent_loc(data_id.loc[sust_vis, value_col], # worsening confirmed at last visit, or first visit after `require_sust_days` days
                                    bl[value_col], type='wors', baseline_delta=data_id.loc[bl_idx, valuedelta_col])

                    # If the event is retained as per `require_sust_days` (i.e., valid_prog==1):
                    # 1. we check if it's PIRA/RAW;
                    # 2. we store the info.
                    if valid_prog:

                        include_event = True  # will become False if searching for specific type of CDW

                        nev = len(event_type)

                        sust_idx = nvisits - 1 if next_nonsust is None else next_nonsust - 1

                        if (data_id.loc[change_idx, 'closest_rel-'] <= relapse_assoc[0]
                            or data_id.loc[change_idx, 'closest_rel+'] <= relapse_assoc[1]): # event is relapse-associated
                            if event=='firstPIRA' and baseline in ('fixed', 'roving_impr'):
                                # skip this event if only searching for PIRA with no CDW-driven re-baseline
                                if verbose==2:
                                    print('Worsening confirmed but not a PIRA event: skipped')
                                include_event = False
                            else:
                                event_type.append('RAW')
                                event_index.append(change_idx)
                        else: # event is not relapse-associated
                            if event == 'firstRAW' and baseline in ('fixed', 'roving_impr'):
                                # skip this event if only searching for RAW with no CDW-driven re-baseline
                                if verbose == 2:
                                    print('Worsening confirmed but not a RAW event: skipped')
                                include_event = False
                            else:
                                # Check if it's PIRA *(

                                if relapse_indep['prec_type'] == 'baseline':
                                    prec = bl
                                elif relapse_indep['prec_type'] == 'last':
                                    prec = data_id.loc[change_idx - 1, :]  # last visit before the event
                                else:
                                    valid_ref = False
                                    iref = change_idx
                                    while iref > 1 and not valid_ref:
                                        iref = iref - 1
                                        if skip_local_extrema != 'none':
                                            prec = data_id.loc[iref, :] if iref == 0 else data_id.loc[iref - 1, :]
                                            prec2value = -1 if iref <= 1 else data_id.loc[iref - 2, value_col]
                                            subs = data_id.loc[iref, :] if iref == nvisits - 1 else data_id.loc[iref + 1, :]
                                            vis = data_id.loc[iref, :]
                                            local_extr = ((
                                                isevent_loc(prec[value_col], vis[value_col], type='wors',
                                                                       st=skip_local_extrema == 'all', baseline_delta=vis[valuedelta_col])
                                                and isevent_loc(subs[value_col], vis[value_col], type='wors',
                                                            st=skip_local_extrema == 'all', baseline_delta=vis[valuedelta_col])
                                                           ) or (
                                                isevent_loc(prec[value_col], vis[value_col], type='impr',
                                                        st=skip_local_extrema == 'all', baseline_delta=vis[valuedelta_col])
                                                and isevent_loc(subs[value_col], vis[value_col], type='impr',
                                                            st=skip_local_extrema == 'all', baseline_delta=vis[valuedelta_col])
                                            )) and (vis[value_col] != prec2value)
                                        else:
                                            local_extr = False
                                        #
                                        event_ok = isevent_loc(data_id.loc[change_idx, value_col], data_id[iref, value_col],
                                                               type='wors', baseline_delta=data_id[iref, valuedelta_col])
                                        conf_ok = (all([isevent_loc(data_id.loc[x, value_col], data_id[iref, value_col], type='wors',
                                                    baseline_delta=data_id.loc[iref, valuedelta_col])
                                                     for x in range(change_idx + 1, conf_idx[0] + 1)])  # worsening is confirmed at (all visits up to) first valid date
                                                if check_intermediate else isevent_loc(data_id.loc[conf_idx[0],value_col], data_id[iref, value_col],
                                                    type='wors', baseline_delta=data_id.loc[iref, valuedelta_col]))  # worsening is confirmed at first valid date
                                        valid_ref = event_ok and conf_ok and not local_extr

                                    prec = data_id.loc[iref, :] if valid_ref else bl  # last pre-worsening visit

                                intervals = {ic : [] for ic in conf_idx}
                                for ic in conf_idx:
                                    for point in ('prec', 'event', 'conf'):
                                        t = prec[date_col] if point=='prec' else (data_id.loc[change_idx,date_col]
                                                if point=='event' else data_id.loc[ic,date_col])
                                        if relapse_indep[point][0] is not None:
                                            t0 = t - relapse_indep[point][0]
                                        if relapse_indep[point][1] is not None:
                                            t1 = t + relapse_indep[point][1]
                                            if t1>t0:
                                                intervals[ic].append([t0,t1])
                                rel_inbetween = [np.logical_or.reduce([(a[0]<=relapse_dates) & (relapse_dates<=a[1])
                                                for a in intervals[ic]]).any() for ic in conf_idx]

                                # Store info on PIRA:
                                pconf_idx = [conf_idx[i] for i in range(len(conf_idx)) if not rel_inbetween[i]]  #_piraconf_#
                                # pconf_idx = conf_idx if not any(rel_inbetween) else conf_idx[:next(i for i in
                                #                                         range(len(conf_idx)) if rel_inbetween[i])]
                                pconf_t = copy.deepcopy(conf_t) #_conf_# [conf_t[i] for i in range(len(conf_t)) if not rel_inbetween[i]] #conf_t[:len(pconf_idx)] # #_piraconf_#
                                if len(pconf_idx)>0: # if pira is confirmed
                                    ######## #_conf_#
                                    for m in conf_days:
                                        confirmed_at = np.intersect1d(pconf_t[m], pconf_idx)
                                        if len(confirmed_at)==0:
                                            del pconf_t[m]
                                        pira_conf[m].append(int(len(confirmed_at)>0))
                                    ######## #_conf_#

                                    event_type.append('PIRA')
                                    event_index.append(change_idx)
                                elif event=='firstPIRA' and baseline in ('fixed', 'roving_impr'):  # if pira is not confirmed, and we're not using it for rebaseline
                                    if verbose==2:
                                        print('Worsening confirmed but not a PIRA event: skipped')
                                    include_event = False
                                else:
                                    event_type.append('undef_CDW')
                                    event_index.append(change_idx)
                                # )*

                        # Store info
                        if include_event:
                            # **(
                            if event_type[-1] != 'PIRA':
                                for m in conf_days:
                                    pira_conf[m].append(None)

                            bldate.append(global_start + datetime.timedelta(days=bl[date_col].item())) #_d_#
                            blvalue.append(bl[value_col])
                            edate.append(global_start + datetime.timedelta(days=data_id.loc[change_idx,date_col].item())) #_d_#
                            evalue.append(data_id.loc[change_idx,value_col])
                            time2event.append(data_id.loc[change_idx,date_col] - data_id.loc[0,date_col]) #.days #_d_#
                            bl2event.append(data_id.loc[change_idx,date_col] - bl[date_col]) #.days #_d_#
                            for m in conf_days:
                                confirmed_at = np.intersect1d(conf_t[m], conf_idx)
                                if len(confirmed_at)==0:
                                    del conf_t[m]
                                conf[m].append(1 if len(confirmed_at) > 0 else 0) #_conf_# 1 if m in conf_t else 0
                            sustd.append(data_id.loc[sust_idx,date_col] - data_id.loc[change_idx,date_col]) #.days #_d_#
                            sustl.append(int(sust_idx == nvisits-1))

                            # Print info
                            if verbose == 2:
                                print('%s %s (visit no.%d, %s) confirmed at %s weeks, sustained up to visit no.%d (%s)'
                                      %(outcome.upper(), event_type[-1], change_idx+1,
                                        global_start + datetime.timedelta(days=data_id.loc[change_idx,date_col].item()), #_d_#
                                        ', '.join([str(x) for x in (pconf_t.keys() if event_type[-1]=='PIRA'
                                                                    else conf_t.keys())]), #_conf_#
                                        sust_idx+1,
                                        global_start + datetime.timedelta(days=data_id.loc[sust_idx,date_col].item()))) #_d_#
                            # )**

                    else:
                        # If the event is NOT retained as per `require_sust_days` (i.e., valid_prog==0):, we proceed.
                        include_event = False
                        for m in conf_days:
                            confirmed_at = np.intersect1d(conf_t[m], conf_idx)
                            if len(confirmed_at) == 0:
                                del conf_t[m]
                        if verbose == 2:
                            print('Change confirmed but not sustained for >=%d days: proceed with search'
                                  %require_sust_days)

                    if len(conf_t)>0:
                        # For each m in conf_days, only keep the earliest available confirmation visit
                        conf_idx = [np.min([x for x in conf_t[m] if x in conf_idx]) for m in conf_t.keys()]

                        if baseline in ('roving', 'roving_wors'):
                            # In a roving baseline setting, the baseline is moved after the confirmed event (even if it is not sustained):
                            newref = conf_idx[0] if proceed_from == 'firstconf' else change_idx
                            bl_idx = newref
                        elif not include_event:
                            # If the event is NOT retained (as per `require_sust_days`, or != required event type),
                            # proceed with search starting from event, regardless of `proceed_from`:
                            newref = change_idx
                        else:
                            # If baseline is static, only update `newref` (to be used for search index):
                            newref = conf_idx[0] if proceed_from == 'firstconf' else change_idx

                        # Move the search index.
                        search_idx = newref + 1

                    else:  # worsening occurring at last visit (len(conf_t)==0)
                        search_idx = change_idx + 1

                    if verbose == 2 and baseline != 'fixed':
                        print(f'Baseline at visit no.{bl_idx + 1}')


                # Confirmed sub-threshold CDW: RE-BASELINE
                # -----------------------------------------
                elif (len(conf_idx) > 0 # confirmation visits available
                        and data_id.loc[change_idx,value_col] > bl[value_col] # value increased from baseline
                        and (all([data_id.loc[x,value_col] > bl[value_col]
                                 for x in range(change_idx+1,conf_idx[0]+1)]) # increase is confirmed
                        if check_intermediate else data_id.loc[conf_idx[0],value_col] > bl[value_col])
                        and baseline in ('roving', 'roving_wors') and sub_threshold_rebl in ('event', 'worsening')
                        ):

                    newref = conf_idx[0] if proceed_from == 'firstconf' else change_idx

                    # Set new baseline after the event:
                    bl_idx = newref
                    # Move search index
                    search_idx = newref + 1

                    if verbose == 2:
                        print(f'Confirmed sub-threshold {outcome.upper()} worsening (visit no.{change_idx + 1})')
                        print(f'Baseline at visit no.{bl_idx + 1}')

                # NO confirmation:
                # ----------------
                else:
                    search_idx = change_idx + 1 # skip the change and look for other patterns after it
                    if verbose == 2:
                        print('Change not confirmed: proceed with search')

            # Relapse-based rebaseline: if search_idx crossed a relapse, move baseline after it.
            # (unless search_idx is after the relapse but within its influence as per `relapse_assoc[0]`: could be a RAW)
            if (relapse_rebl and proceed and search_idx < nvisits
                and ((data_id.loc[bl_idx, date_col] < relapse_dates)  # presence of a relapse between baseline...
                 & (relapse_dates + (0 if event=='firstPIRA' else relapse_assoc[0])
                    < data_id.loc[search_idx, date_col]  # ...and search index
                    )).any()
                ):
                proceed = 1

                # Move baseline just after `irel`-th relapse
                bl_idx = next((x for x in range(bl_idx, nvisits) # visits from current baseline
                               if relapse_dates[irel] <= data_id.loc[x, date_col]  # after `irel`-th relapse
                               ),
                              None)

                # Move search index just after baseline
                if bl_idx is not None:
                    search_idx = bl_idx + 1
                    if verbose == 2:
                        print(f'[post-relapse rebaseline] Baseline at visit no.{bl_idx + 1}')

                # If no more rebaseline is possible, terminate search:
                if proceed and (bl_idx is None or bl_idx > nvisits - 2):
                    proceed = 0
                    if verbose == 2:
                        print('[post-relapse rebaseline] Not enough visits after current baseline: end process')

            if proceed and (
                (event == 'first' and len(event_type)>1)
                or (event == 'firsteach' and ('impr' in event_type) and (('RAW' in event_type) or ('PIRA' in event_type) or ('undef_CDW' in event_type)))
                or (event == 'firstCDW' and (('RAW' in event_type) or ('PIRA' in event_type) or ('undef_CDW' in event_type)))
                or (event == 'firstCDWtype' and ('RAW' in event_type) and ('PIRA' in event_type))  # and ('undef_CDW' in event_type)
                or (event == 'firstPIRA' and ('PIRA' in event_type))
                or (event == 'firstRAW' and ('RAW' in event_type))
                        ):
                    proceed = 0
                    if verbose == 2:
                        print('\'%s\' events already found: end process' %event)

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
                prog_idx = next((x for x in range(len(event_type)) if event_type[x] in ('undef_CDW', 'RAW', 'PIRA')), None)
                raw_idx = next((x for x in range(len(event_type)) if event_type[x]=='RAW'), None)
                pira_idx = next((x for x in range(len(event_type)) if event_type[x]=='PIRA'), None)
                undef_prog_idx = next((x for x in range(len(event_type)) if event_type[x]=='undef_CDW'), None)
                if event=='firsteach':
                    first_events = [impr_idx, prog_idx]
                elif event=='firstCDW':
                    first_events = [prog_idx]
                elif event=='firstCDWtype':
                    first_events = [raw_idx, pira_idx]  # , undef_prog_idx
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
                for m in conf_days:
                    results.loc[results[subj_col]==subjid, 'conf'+str(m)] = [conf[m][i] for i in event_order]
                results.loc[results[subj_col]==subjid, 'sust_days'] = [sustd[i] for i in event_order]
                results.loc[results[subj_col]==subjid, 'sust_last'] = [sustl[i] for i in event_order]
                for m in conf_days: #[1:]: #_piraconf_#
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
        CDW = results.loc[results[subj_col]==subjid, 'event_type'].isin(('undef_CDW', 'RAW', 'PIRA')).sum()
        undefined_CDW = (results.loc[results[subj_col]==subjid, 'event_type']=='undef_CDW').sum()
        RAW = (results.loc[results[subj_col]==subjid, 'event_type']=='RAW').sum()
        PIRA = (results.loc[results[subj_col]==subjid, 'event_type']=='PIRA').sum()
        summary.loc[subjid, ['event_sequence', 'improvement', 'CDW',
                'RAW', 'PIRA', 'undefined_CDW']] = [', '.join(event_type), improvement, CDW,
                                                     RAW, PIRA, undefined_CDW]
        if event.startswith('firstCDW'):
            summary.drop(columns=['improvement'], inplace=True)

        if verbose == 2:
            print('Event sequence: %s' %(', '.join(event_type) if len(event_type)>0 else '-'))

    if verbose>=1:
        print(f'''\n---\nOutcome: {outcome.upper()}\nConfirmation {"over" if check_intermediate else "at"}: \
{conf_days} days (-{conf_tol_days[0]} days, +{"inf" if conf_tol_days[1]==np.inf else str(conf_tol_days[1])+" days"})
Baseline: {baseline}{" (including sub-threshold %s)" %sub_threshold_rebl if baseline!='fixed' and sub_threshold_rebl!="none" else ""}\
{" (and post-relapse re-baseline)" if relapse_rebl else ""}
Relapse influence (baseline): {relapse_to_bl} days\nRelapse influence (event): {relapse_to_event} days
Relapse influence (confirmation): {relapse_to_conf} days\nEvents detected: {event}
        ''')
        print('---\nTotal subjects: %d\n---\nSubjects with disability worsening: %d (PIRA: %d; RAW: %d)'
              %(nsub, (summary['CDW']>0).sum(),
                (summary['PIRA']>0).sum(), (summary['RAW']>0).sum()))
        if event not in ('firstCDW', 'firstCDWtype', 'firstPIRA', 'firstRAW'):
            print('Subjects with disability improvement: %d' %(summary['improvement']>0).sum())
        if event in ('multiple', 'firstCDWtype'):
            print('---\nCDW events: %d (PIRA: %d; RAW: %d)'
                  %(summary['CDW'].sum(),
                    summary['PIRA'].sum(), summary['RAW'].sum()))
        if event=='multiple':
            print('Improvement events: %d' %(summary['improvement'].sum()))

    columns = results.columns
    if not include_dates:
        columns = [c for c in columns if not c.endswith('date')]
    if not include_value:
        columns = [c for c in columns if not c.endswith('value')]

    scolumns = summary.columns
    if event == 'firstPIRA':
        scolumns = ['PIRA']
    elif event == 'firstRAW':
        scolumns = ['RAW']
        columns = [c for c in columns if not c.startswith('PIRA')]
    elif event in ('firstCDW', 'firstCDWtype'):
        scolumns = [c for c in scolumns if c != 'improvement']

    summary = summary[scolumns]
    results = results[columns]

    for w in warning_msgs:
        warnings.warn(w)

    return summary, results.reset_index(drop=True)


#####################################################################################

def compute_delta(baseline, outcome='edss'):
    '''
    Definition of progression deltas for different tests.

    Parameters
    ----------
    baseline: float
        Baseline value.
    outcome: str
        Type of test ('edss'[default],'nhpt','t25fw','sdmt')

    Returns
    -------
    float
        Minimum delta corresponding to valid change in the outcome measure.

    '''
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
    '''

    Check for change from baseline.

    Parameters
    ----------
     x: float
        New value.
     baseline: float
        Baseline value.
     type: str
        'wors' or 'impr' or 'change'.
     outcome: str
        Type of test (one of: 'edss','nhpt','t25fw','sdmt').
     worsening: 'increase' or 'decrease'. If outcome is specified, it is automatically assigned
                ('increase' for edss, nhpt, t25fw; 'decrease' for sdmt)
     baseline_delta: float
        Baseline value to use to compute delta, if different from baseline.

    Returns
    -------
    bool
        Whether `x` is an event with respect to `baseline`.
    '''
    if baseline_delta is None:
        baseline_delta = baseline

    if outcome in ('edss', 'nhpt', 't25fw'):
        worsening = 'increase'
    elif outcome=='sdmt':
        worsening = 'decrease'
    elif worsening is None:
        raise ValueError('Either specify a valid outcome type, or specify worsening direction')
    improvement = 'increase' if worsening=='decrease' else 'decrease'

    if sub_threshold:
        event_sign = {'increase': x > baseline, 'decrease': x < baseline, 'change': x != baseline}
    else:
        if delta_fun is None and outcome is None:
            raise ValueError('Either specify a valid outcome type, or specify a custom `delta_fun`')
        elif delta_fun is None:
            fun_tmp = compute_delta
        else:
            def fun_tmp(baseline, outcome):
                try:
                    return delta_fun(baseline, outcome)
                except TypeError:
                    return delta_fun(baseline)
        event_sign = {'increase': x - baseline >= fun_tmp(baseline_delta, outcome),
                      'decrease': x - baseline <= - fun_tmp(baseline_delta, outcome),
                      'change': abs(x - baseline) >= fun_tmp(baseline_delta, outcome)}
    event = {'wors': event_sign[worsening], 'impr': event_sign[improvement], 'change': event_sign['change']}
    return event[type]


#####################################################################################

def value_milestone(data, milestone, value_col, date_col, subj_col,
                   relapse=None, rsubj_col=None, rdate_col=None,
                   validconf_col=None, conf_days=7*12, conf_tol_days=[7, 2*365.25],
                   relapse_to_event=0, relapse_to_conf=30,
                   verbose=0):
    '''

    Time to disability milestone.

    The function scans the visits in chronological order to detect the first
    outcome value exceeding a specified disability milestone (e.g., EDSS>=6), *with confirmation*.
    Note: "exceeding" means either value>=milestone or value<=milestone, depending on the
    outcome measure (see arguments ``outcome`` and ``worsening``).

    Parameters
    ----------
    data: pandas.DataFrame
        Longitudinal data containing subject ID, outcome value, date of visit.

    milestone: float
        Disability milestone (outcome value to check data against).

    subj_col: str
        Name of data column with subject ID.

    value_col: str
        Name of data column with outcome value.

    date_col: str
        Name of data column with date of visit.

    relapse: pandas.DataFrame
        Longitudinal data containing subject ID and relapse date (can be omitted).

    rsubj_col: str
        Name of subject ID column for relapse data, if different from outcome data.

    rdate_col: str
        Name of onset date column for relapse data, if different from outcome data.

    validconf_col: str
        Name of data column specifying which visits can (True) or cannot (False) be used as confirmation visits.
        The input data does not necessarily have to include such a column.
        If not specified (``validconf_col=None``), all visits are potentially used as confirmation visits.

    conf_days: int, float, or array-like
        Period before confirmation (days). Can be a single value or array-like of any length if multiple
        windows are to be considered.

    conf_tol_days: int, float, or array-like of length 1 or 2
        Tolerance window for confirmation visit (days); can be a single value (same tolerance on left and right)
        or array-like of length 2 (different tolerance on left and right).
        The right end of the interval can be set to ``float('inf')`` (confirmation window unbounded on the right
        -- e.g., "confirmed over 12 *or more* weeks").

    relapse_to_event: int, float, or array-like of length 1 or 2
        Minimum distance from a relapse (days) for an event to be considered as such.
        Can be a single value (minimum distance from *last* relapse) or array-like of length 2
        (minimum distance from *last* relapse, minimum distance from *next* relapse).
        Note that setting the distance to zero means retaining the event regardless of surrounding relapses.
        # If relapse end dates are available (``renddate_col``), the minimum distance from last relapse
        # is overwritten by the relapse duration, unless it was set to zero (in which case it stays 0).

    relapse_to_conf: int, float, or array-like of length 1 or 2
        Minimum distance from a relapse (days) for a visit to be a valid confirmation visit.
        Can be a single value (minimum distance from *last* relapse) or array-like of length 2
        (minimum distance from *last* relapse, minimum distance from *next* relapse).
        Note that setting the distance to zero means using any visit for confirmation regardless of surrounding relapses.
        # If relapse end dates are available (``renddate_col``), the minimum distance from last relapse
        # is overwritten by the relapse duration, unless it was set to zero (in which case it stays 0).

    verbose: int
        One of: 0 (print no info); 1 (print progress info).

    Returns
    -------
    pandas.DataFrame
        Including columns:

        - ``date_col`` -- date of first reaching value >=milestone (or last date of follow-up if milestone is not reached);
        - ``value_col`` -- first value >=milestone, if present, otherwise last value recorded.

    Notes
    -----
    An event is only retained if **confirmed**, i.e., if all values *up to* the
    confirmation visit exceed the milestone.
    Valid time windows for confirmation visits are determined by arguments
    ``conf_days``, ``conf_tol_days``, ``relapse_to_conf``.

    Raises
    ------
    ValueError
        If the arguments are incorrectly specified.

    '''

    ###########################
    # CHECKS ON ARGUMENT VALUES

    # If conf_days is a single value, make it a list with a single element
    try:
        _ = (e for e in conf_days)  # check if conf_days is iterable
    except TypeError:
        conf_days = [conf_days]

    # If conf_tol_days is a single value, duplicate it (equal left and right tolerance)
    try:
        _ = (e for e in conf_tol_days)  # check if it's iterable
        if len(conf_tol_days) == 1:
            conf_tol_days = [conf_tol_days[0], conf_tol_days[0]]
    except TypeError:
        conf_tol_days = [conf_tol_days, conf_tol_days]

    # If relapse_to_event is a single value, set right bound to zero
    try:
        _ = (e for e in relapse_to_event)  # check if it's iterable
        if len(relapse_to_event) == 1:
            relapse_to_event = [relapse_to_event[0], 0]
    except TypeError:
        relapse_to_event = [relapse_to_event, 0]

    # If relapse_to_conf is a single value, set right bound to zero
    try:
        _ = (e for e in relapse_to_conf)  # check if it's iterable
        if len(relapse_to_conf) == 1:
            relapse_to_conf = [relapse_to_conf[0], 0]
    except TypeError:
        relapse_to_conf = [relapse_to_conf, 0]

    # end of checks
    ###########################

    # If no column names are specified for the relapse file, use the main ones
    if relapse is not None and rsubj_col is None:
        rsubj_col = subj_col
    if relapse is not None and rdate_col is None:
        rdate_col = date_col

    # If no `validconf_col` is specified, create a dummy one
    if validconf_col is None:
        validconf_col = 'validconf'
        data.loc[:, validconf_col] = 1

    # Remove missing values from columns of interest
    data = data[[subj_col, value_col, date_col, validconf_col]].dropna()

    # Convert dates to datetime.date format
    data[date_col] = pd.to_datetime(data[date_col])
    if relapse is None:
        relapse = pd.DataFrame([], columns=[rsubj_col, rdate_col])
        relapse_start = data[date_col].min()
    else:
        relapse = relapse[[rsubj_col, rdate_col]].copy().dropna() # remove missing values from columns of interest
        relapse[rdate_col] = pd.to_datetime(relapse[rdate_col])
        relapse_start = relapse[rdate_col].min()
    # Convert dates to days from minimum #_d_#
    global_start = min(data[date_col].min(), relapse_start)
    relapse[rdate_col] = (relapse[rdate_col] - global_start).apply(lambda x : x.days)
    data[date_col] = (data[date_col] - global_start).apply(lambda x : x.days)

    conf_window = [(int(c) - conf_tol_days[0], int(c) + conf_tol_days[1]) for c in conf_days]

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
        if verbose:
            print('\nSubject #%s: %d visit%s'
              %(subjid,nvisits,'' if nvisits==1 else 's'))
            if any(ucounts>1):
                print('Found multiple visits in the same day: only keeping last')
        first_visit = data_id[date_col].min()
        if relapse is not None:
            relapse_id = relapse.loc[relapse[rsubj_col]==subjid,:].reset_index(drop=True)
            relapse_id = relapse_id.loc[relapse_id[rdate_col] >= first_visit - relapse_to_event[0], :]
                                                        # ignore relapses occurring before first visit
            relapse_dates = relapse_id[rdate_col].values
            relapse_df = pd.DataFrame([relapse_dates]*len(data_id))
            relapse_df['visit'] = data_id[date_col].values
            dist = relapse_df.drop(['visit'],axis=1).subtract(relapse_df['visit'], axis=0) #_d_# .apply(lambda x : pd.to_timedelta(x).dt.days)
            distm = - dist.mask(dist > 0)  # other=-float('inf')
            distp = dist.mask(dist < 0)  # other=float('inf')
            distm[distm.isna()] = float('inf')
            distp[distp.isna()] = float('inf')
            data_id['closest_rel-'] = float('inf') if all(distm.isna()) else distm.min(axis=1)
            data_id['closest_rel+'] = float('inf') if all(distp.isna()) else distp.min(axis=1)
        else:
            data_id['closest_rel-'] = float('inf')
            data_id['closest_rel+'] = float('inf')

        proceed = 1
        search_idx = 0
        while proceed:
            milestone_idx = next((x for x in range(search_idx,nvisits)
                    if data_id.loc[x,value_col]>=milestone  # first occurring value>=milestone
                    and (data_id.loc[x, 'closest_rel-'] >= relapse_to_event[0])  # out of influence of last relapse
                    and (data_id.loc[x, 'closest_rel+'] >= relapse_to_event[1])  # out of influence of next relapse
                                  ), None)
            if milestone_idx is None: # value does not change in any subsequent visit
                results.at[subjid,date_col] = global_start + datetime.timedelta(days=data_id.iloc[-1,:][date_col].item()) #_d_# data_id.iloc[-1,:][date_col]
                results.at[subjid,value_col] = data_id.iloc[-1,:][value_col]
                proceed = 0
                if verbose:
                    print('No value >=%d in any visit: end process' %(milestone))
            else:
                conf_idx = [[x for x in range(milestone_idx+1, nvisits)
                        if c[0] <= data_id.loc[x,date_col] - data_id.loc[milestone_idx,date_col] <= c[1] # date in confirmation range
                        and data_id.loc[x,'closest_rel-'] >= relapse_to_conf[0] # out of influence of last relapse
                        and data_id.loc[x,'closest_rel+'] >= relapse_to_conf[1] # out of influence of last relapse
                        and data_id.loc[x, validconf_col]]  # can be used as confirmation
                        for c in conf_window]
                conf_idx = np.unique([x for i in range(len(conf_idx)) for x in conf_idx[i]])
                if len(conf_idx)>0 and all([data_id.loc[x,value_col]
                            >= milestone for x in range(milestone_idx+1,conf_idx[0]+1)]):
                    results.at[subjid,date_col] = global_start + datetime.timedelta(days=data_id.loc[milestone_idx,date_col].item()) #_d_# #data_id.loc[milestone_idx,date_col]
                    results.at[subjid,value_col] = data_id.loc[milestone_idx,value_col]
                    proceed = 0
                    if verbose:
                        print('Found value >=%d: end process' %(milestone))
                else:
                    next_change = next((x for x in range(milestone_idx+1,nvisits)
                    if data_id.loc[x,value_col]<milestone), None)
                    search_idx = search_idx + 1 if next_change is None else next_change + 1
                    if verbose:
                        print('value >=%d not confirmed: continue search' %(milestone))

        if results.at[subjid,date_col] is None:
            results.at[subjid, date_col] = global_start + datetime.timedelta(days=data_id.iloc[-1,:][date_col] .item()) #_d_# data_id.iloc[-1,:][date_col]
            results.at[subjid, value_col] = data_id.iloc[-1,:][value_col]

    return results


#####################################################################################


def separate_ri_ra(data, subj_col, value_col, date_col, outcome, relapse,
                   rsubj_col=None, rdate_col=None, delta_fun=None, worsening=None,
                   validconf_col=None, conf_days=7*12, conf_tol_days=[7, 2*365.25], sust_raw_days=180,
                   relapse_to_bl=30, relapse_to_conf=30, relapse_assoc=90, impute_last_visit=0,
                   subtract_bl=False, include_rel_num=False, include_bl=True,
                   include_raw_dates=False, verbose=0):
    '''

    Given longitudinal disability assessments and relapse dates, decompose
    the disability trajectory into a relapse-associated component (RAC)
    and a relapse-independent component (RIC).

    Parameters
    ----------
    data: pandas.DataFrame
        Longitudinal data containing subject ID, outcome value, date of visit.

    subj_col: str
        Name of data column with subject ID.

    value_col: str
        Name of data column with outcome value.

    date_col: str
        Name of data column with date of visit.

    outcome: str
        Outcome type. Must be one of the following:

        - 'edss' (Expanded Disability Status Scale)
        - 'nhpt' (Nine-Hole Peg Test)
        - 't25fw' (Timed 25-Foot Walk)
        - 'sdmt' (Symbol Digit Modalities Test)
        - None (only accepted when specifying a custom ``delta_fun``)

    relapse: pandas.DataFrame
        Longitudinal data containing subject ID and relapse date.

    rsubj_col: str
        Name of subject ID column for relapse data, if different from outcome data.

    rdate_col: str
        Name of onset date column for relapse data, if different from outcome data.

    delta_fun: function
        Custom function specifying the minimum shift in the outcome measure that counts as a valid change
        from the current reference value.
        The function provided must take a numeric value (reference score) as input, and return a numeric
        value corresponding to the minimum shift from baseline.
        If none is specified (default), the user must provide a non-None value for the ``outcome`` argument
        in order to use the built-in function :func:`compute_delta()`.

    worsening: str
        The direction of worsening ('increase' if higher values correspond to worse disease course, 'decrease' otherwise).
        This argument is only used when ``outcome`` is set to None. Otherwise, ``worsening`` is automatically set to
        'increase' if ``outcome`` is set to 'edss', 'nhpt', 't25fw', and to 'decrease' if ``outcome`` is set to 'sdmt'.

    validconf_col: str
        Name of data column specifying which visits can (True) or cannot (False) be used as confirmation visits.
        The input data does not necessarily have to include such a column.
        If not specified (``validconf_col=None``), all visits are potentially used as confirmation visits.

    conf_days: int, float, or array-like
        Period before confirmation (days). Can be a single value or array-like of any length if multiple
        windows are to be considered.

    conf_tol_days: int, float, or array-like of length 1 or 2
        Tolerance window for confirmation visit (days); can be a single value (same tolerance on left and right)
        or array-like of length 2 (different tolerance on left and right).
        The right end of the interval can be set to ``float('inf')`` (confirmation window unbounded on the right
        -- e.g., "confirmed over 12 *or more* weeks").

    sust_raw_days: int or float
        Minimum number of days over which a confirmed RAW event must be maintained
        (i.e., confirmed at *all* visits occurring in the specified period) to be considered as "sustained RAW".
        Events confirmed for the remainder of the follow-up period are always considered sustained
        even if follow-up duration is shorter than ``sust_raw_days``.
        Setting ``sust_raw_days=float('inf')``, events are considered sustained only if the change is maintained
        for the remainder of the follow-up period.

    relapse_to_bl: int, float, or array-like of length 1 or 2
        Minimum distance from a relapse (days) for a visit to be used as baseline.
        Can be a single value (minimum distance from *last* relapse) or array-like of length 2
        (minimum distance from *last* relapse, minimum distance from *next* relapse).
        Note that setting the distance to zero means keeping the baseline where it is regardless of surrounding relapses.
        # If relapse end dates are available (``renddate_col``), the minimum distance from last relapse
        # is overwritten by the relapse duration, unless it was set to zero (in which case it stays 0).
        If the designated baseline does not respect this constraint, the baseline is moved to the next available visit.

    relapse_to_conf: int, float, or array-like of length 1 or 2
        Minimum distance from a relapse (days) for a visit to be a valid confirmation visit.
        Can be a single value (minimum distance from *last* relapse) or array-like of length 2
        (minimum distance from *last* relapse, minimum distance from *next* relapse).
        Note that setting the distance to zero means using any visit for confirmation regardless of surrounding relapses.
        # If relapse end dates are available (``renddate_col``), the minimum distance from last relapse
        # is overwritten by the relapse duration, unless it was set to zero (in which case it stays 0).

    relapse_assoc: int, float, or array-like of length 1 or 2
        Maximum distance from a relapse (days) for a CDW event to be classified as RAW.
        Can be a single value (maximum distance from *last* relapse) or array-like of length 2
        (maximum distance from *last* relapse, maximum distance from *next* relapse).
        # If relapse end dates are available (``renddate_col``), the maximum distance from last relapse
        # is overwritten by the relapse duration.

    impute_last_visit: float or int
        Imputation probability for RAW events occurring at last visit (i.e. with no confirmation).
        Unconfirmed worsening events occurring at the last visit are never imputed if ``impute_last_visit=0``;
        they are always imputed if ``impute_last_visit=1``;
        they are imputed with probability ``p``, ``0<p<1``, if ``impute_last_visit=p``.
        If a value ``N>1`` is passed, unconfirmed worsening events are imputed only if occurring within
        ``N`` days of follow-up (e.g., in case of early discontinuation).

    subtract_bl: bool
        If True, the returned DataFrame will also include (``f'{value_col}-bl'`` column) the overall change
        in score from baseline (simply obtained by subtracting the overall baseline from ``data[value_col]``).

    include_rel_num: bool
        If True, the returned DataFrame will include the cumulative number of relapses ('relapse_num' column)
        at all visits of each subject.

    include_bl: bool
        If True, the returned DataFrame will include the baseline outcome value (``f'bl{value_col}'`` column)
        for each subject.

    include_raw_dates: bool
        If True, an additional DataFrame is returned containing the dates of detected RAW events
        for each subject.

    verbose: int
        One of: 0 (print no info); 1 (print concise info, default); 2 (print extended info).

    Returns
    -------
        pandas.DataFrame or (pandas.DataFrame, pandas.DataFrame)
        The original DataFrame (``data``) is returned with the addition of the following columns:

        - ``f'ric_{value_col}'``: cumulative relapse-independent change from baseline
        - ``f'srac_{value_col}'``: cumulative sustained relapse-associated change from baseline
        - ``f'bumps_{value_col}'``: transient relapse-associated changes (bumps)
        - ``f'bl_{value_col}'``: baseline score (if ``include_bl=True``)
        - 'relapse_num': cumulative relapse number (if ``include_rel_num=True``)

        If ``include_raw_dates=True``, an additional DataFrame containing the dates of detected
        RAW events for each subject is also returned.

    Notes
    -----
    For each subject, the function analyses the disability assessments surrounding each relapse
    to look for a *confirmed* worsening in the disability score within the influence period of
    the relapse (as per ``relapse_assoc[0]``) with respect to the last visit preceding the relapse
    (at a distance specified by ``relapse_assoc[1]``). If such a worsening is found,
    it is labelled as a relapse-associated worsening (RAW) event.

    The change in score resulting from each RAW event is subtracted from the overall trajectory to isolate PIRA.
    If a RAW event is sustained for a specified period of time (as per ``sust_raw_days`` argument),
    the accumulation of disability is considered permanent and the change is subtracted from all subsequent
    visits. The change is computed as the minimum score over the sustained period; any temporary
    worsening exceeding the change is considered as a "bump", and is only subtracted from the score of
    the current assessment. If a RAW event is transient (not sustained), the change is only
    subtracted from all scores up to the confirmation visit.
    Three sub-trajectories are obtained:

    - RAC (permanent accumulation of disability from sustained RAW);
    - "bumps" (temporary disability worsening from transient RAW);
    - RIC (disability accumulation due to PIRA, obtained as the trajectory of disability worsening
      deprived of all sustained and transient relapse-associated contributions).

    All sub-trajectories are expressed as deltas relative to the baseline value,
    as the disability already accumulated at the beginning of the follow-up period
    cannot be attributed to relapse-associated or relapse-independent CDW
    with the available data.

    Raises
    ------
    ValueError
        If the arguments are incorrectly specified.

    '''

    ###########################
    # CHECKS ON ARGUMENT VALUES

    # If conf_days is a single value, make it a list with a single element
    try:
        _ = (e for e in conf_days)  # check if conf_days is iterable
    except TypeError:
        conf_days = [conf_days]

    # If conf_tol_days is a single value, duplicate it (equal left and right tolerance)
    try:
        _ = (e for e in conf_tol_days)  # check if it's iterable
        if len(conf_tol_days) == 1:
            conf_tol_days = [conf_tol_days[0], conf_tol_days[0]]
    except TypeError:
        conf_tol_days = [conf_tol_days, conf_tol_days]

    # If relapse_to_bl is a single value, set right bound to zero
    try:
        _ = (e for e in relapse_to_bl)  # check if it's iterable
        if len(relapse_to_bl) == 1:
            relapse_to_bl = [relapse_to_bl[0], 0]
    except TypeError:
        relapse_to_bl = [relapse_to_bl, 0]

    # If relapse_to_conf is a single value, set right bound to zero
    try:
        _ = (e for e in relapse_to_conf)  # check if it's iterable
        if len(relapse_to_conf) == 1:
            relapse_to_conf = [relapse_to_conf[0], 0]
    except TypeError:
        relapse_to_conf = [relapse_to_conf, 0]

    # If relapse_assoc is a single value, set right bound to zero
    try:
        _ = (e for e in relapse_assoc)  # check if it's iterable
        if len(relapse_assoc) == 1:
            relapse_assoc = [relapse_assoc[0], 0]
    except TypeError:
        relapse_assoc = [relapse_assoc, 0]

    if outcome is None or outcome.lower() not in ['edss', 'nhpt', 't25fw', 'sdmt']:
        outcome = 'outcome'
    else:
        outcome = outcome.lower()

    # end of checks
    ###########################

    # If no column names are specified for the relapse file, use the main ones
    if rsubj_col is None:
        rsubj_col = subj_col
    if rdate_col is None:
        rdate_col = date_col

    # If no `validconf_col` is specified, create a dummy one
    if validconf_col is None:
        validconf_col = 'validconf'
        data.loc[:, validconf_col] = 1

    if impute_last_visit < 0:
        raise ValueError('`impute_last_visit` must be nonnegative')
    elif impute_last_visit <= 1:
        # If impute_last_visit is a probability, set no limit to follow-up length (Inf)
        impute_max_fu = float('inf')
    else:
        # If impute_last_visit is a follow-up time, save the value and set probability to 1
        impute_max_fu = impute_last_visit
        impute_last_visit = 1

    if outcome in ('edss', 'nhpt', 't25fw'):
        worsening = 'increase'
    elif outcome == 'sdmt':
        worsening = 'decrease'
    elif worsening is None:
        raise ValueError(
            'Either specify an outcome type, or specify the direction of worsening (\'increase\' or \'decrease\')')

    def isevent_loc(x, bl, type='wors', st=False):
        return is_event(x, bl, type=type, outcome=outcome, worsening=worsening,
                        sub_threshold=st, delta_fun=delta_fun)

    data_sep = data.copy()
    relapse = relapse.copy()

    # Remove missing values from columns of interest
    data_sep = data_sep.dropna(subset=[subj_col, value_col, date_col, validconf_col])
    # Convert dates to datetime.date format
    data_sep[date_col] = pd.to_datetime(data_sep[date_col])
    if relapse is None:
        relapse = pd.DataFrame([], columns=[rsubj_col, rdate_col])
        relapse_start = data_sep[date_col].min()
    else:
        relapse = relapse[[rsubj_col, rdate_col]].copy().dropna() # remove missing values from columns of interest
        relapse[rdate_col] = pd.to_datetime(relapse[rdate_col])
        relapse_start = relapse[rdate_col].min()
    # Convert dates to days from minimum #_d_#
    global_start = min(data[date_col].min(), relapse_start)
    relapse[rdate_col] = (relapse[rdate_col] - global_start).apply(lambda x : x.days)
    data_sep[date_col] = (data_sep[date_col] - global_start).apply(lambda x : x.days)

    ri_col, ra_col, bump_col = 'ric_' + value_col, 'rac_' + value_col, 'bumps_' + value_col
    data_sep[ra_col] = 0.
    data_sep[bump_col] = 0.
    data_sep[ri_col] = data_sep[value_col]
    if include_rel_num:
        data_sep['relapse_num'] = 0
    if include_bl:
        data_sep['bl_' + value_col] = 0.
    if subtract_bl:
        data_sep[value_col+'-bl'] = 0.

    def within_influence(rel, vis, dist0, dist1, included=True):
        if vis == rel and dist0 + dist1 > 0:  # Visit coincides with relapse, and at least one of dist0 and dist1 is >0
            return True
        elif vis - rel > 0:  # Visit after the relapse: use dist0
            if dist0 > 0:
                return vis - rel <= dist0 if included else vis - rel < dist0
            else:
                return False
        else:  # Visit before the relapse: use dist1
            if dist1 > 0:
                return rel - vis <= dist1 if included else rel - vis < dist1
            else: return False

    conf_window = [(int(c) - conf_tol_days[0], int(c) + conf_tol_days[1]) for c in conf_days]

    all_subj = data[subj_col].unique()
    nsub = len(all_subj)

    if include_raw_dates:
        raw_events = []

    for subjid in all_subj:
        data_id = data_sep.loc[data_sep[subj_col]==subjid,:].copy().reset_index(drop=True)
        nvisits = len(data_id)

        relapse_id = relapse.loc[relapse[rsubj_col]==subjid,:].reset_index(drop=True)

        relapse_dates = relapse_id[rdate_col].values
        relapse_df = pd.DataFrame([relapse_dates]*len(data_id))
        relapse_df['visit'] = data_id[date_col].values
        dist = relapse_df.drop(['visit'],axis=1).subtract(relapse_df['visit'], axis=0)
        distm = - dist.mask(dist > 0)  # other=-float('inf')
        distp = dist.mask(dist < 0)  # other=float('inf')
        distm[distm.isna()] = float('inf')
        distp[distp.isna()] = float('inf')
        data_id['closest_rel-'] = distm.min(axis=1)  #float('inf') if all(distm.isna()) else distm.min(axis=1)
        data_id['closest_rel+'] = distp.min(axis=1)  #float('inf') if all(distp.isna()) else distp.min(axis=1)

        # First visit out of relapse influence
        rel_free_bl = next((x for x in range(len(data_id))
                        if data_id.loc[x, 'closest_rel-'] >= relapse_to_bl[0]
                        and data_id.loc[x, 'closest_rel+'] >= relapse_to_bl[1]), None)

        nrel = len(relapse_id) if len(data_id)==0 else sum(relapse_id[rdate_col] >= data_id.loc[0, date_col])
        if verbose == 2:
            print('\nSubject #%s: %d visit%s, %d relapse%s'
              %(subjid, nvisits, '' if nvisits==1 else 's', nrel, '' if nrel==1 else 's'))

        # Set global baseline
        if rel_free_bl is None:
            data_id = data_id.loc[[],:].reset_index(drop=True)
            global_bl = data_id.copy()
            if verbose == 2:
                print('No baseline visits out of relapse influence')
        elif rel_free_bl > 0:
            glob_bl_idx = data_id.loc[:rel_free_bl, :].sort_values(by=value_col).index[0]
            global_bl = data_id.loc[glob_bl_idx, :].copy()
            data_id = data_id.loc[rel_free_bl:,:].reset_index(drop=True)
            # bump = data_id[value_col] - data_id.loc[:rel_free_bl, value_col].min() # values exceeding the minimum up to the baseline
            # data_id.loc[:rel_free_bl-1, bump_col] = data_id.loc[:rel_free_bl-1, bump_col] + bump.loc[:rel_free_bl-1]
            if verbose == 2:
                print('Moving global baseline to first visit out of relapse influence (visit #%d, %s)'
                      %(rel_free_bl + 1, (global_start + datetime.timedelta(
                                days=data_id.loc[0, date_col].item())).date()))
        else:
            global_bl = data_id.loc[0, :].copy()
        nvisits = len(data_id)

        # Ignore relapses occurring before the baseline
        bl_date = data_id[date_col].min() #data_id[date_col].max() if rel_free_bl is None else data_id.loc[rel_free_bl, date_col] #
        relapse_id = relapse_id.loc[relapse_id[rdate_col] > bl_date, :].reset_index(drop=True)
        if verbose == 2 and rel_free_bl is not None and rel_free_bl > 0:
            print('Relapses left to analyse: %d' %len(relapse_id))

        ##########
        visit_dates = data_id[date_col].values
        relapse_df = pd.DataFrame([visit_dates]*len(relapse_id))
        relapse_df['relapse'] = relapse_id[rdate_col].values
        dist = relapse_df.drop(['relapse'],axis=1).subtract(relapse_df['relapse'], axis=0)
        distm = - dist.mask(dist>0)
        distp = dist.mask(dist<0)
        distm[distm.isna()] = float('inf')
        distp[distp.isna()] = float('inf')
        relapse_id['closest_vis-'] = distm.idxmin(axis=1)  #None if all(distm.isna()) else distm.idxmin(axis=1)
        relapse_id['closest_vis+'] = distp.idxmin(axis=1)  #None if all(distp.isna()) else distp.idxmin(axis=1)
        relapse_id.loc[distm.min(axis=1)==np.inf, 'closest_vis-'] = np.nan
        relapse_id.loc[distp.min(axis=1)==np.inf, 'closest_vis+'] = np.nan
        ##########

        delta_raw, raw_dates = [], []
        last_conf = None  # confirmation period of last detected RAW

        for irel in range(len(relapse_id)):

            if last_conf is not None and last_conf >= relapse_id.loc[irel, rdate_col]:
                if verbose == 2:
                    print('Relapse #%d/%d (%s): skipped (falls within confirmation period of last %s change)'
                          %(irel+1, len(relapse_id), (global_start + datetime.timedelta(
                                days=relapse_id.loc[irel, rdate_col].item())).date(), outcome))
                continue  # go to next relapse

            if verbose == 2:
                print('Relapse #%d/%d (%s)' %(irel+1, len(relapse_id),
                        (global_start + datetime.timedelta(
                            days=relapse_id.loc[irel, rdate_col].item())).date()
                                                ))

            # Baseline set to last visit before the relapse and out of its influence (as per `relapse_to_bl`and `relapse_assoc`):
            bl_idx = next((n for n in range(relapse_id.loc[irel, 'closest_vis-'], -1, -1)  # back from closest visit before relapse
                        if relapse_id.loc[irel, rdate_col] - data_id.loc[n, date_col] > max(relapse_to_bl[1], relapse_assoc[1])),
                        None)
            # bl_idx = next((int(relapse_id.loc[irel, 'closest_vis-'] - n) for n in range(int(relapse_id.loc[irel, 'closest_vis-'])) if
            #                relapse_id.loc[irel, rdate_col] - data_id.loc[int(relapse_id.loc[irel, 'closest_vis-']) - n, date_col] > relapse_to_bl[1]),
            #               None)
            bl = global_bl if bl_idx is None else data_id.loc[bl_idx, :].copy()

            # If baseline is part of a bump caused by a previous relapse, subtract the bump
            # (unless it ends up below global baseline):
            bl[value_col] = max(bl[value_col] - bl[bump_col], global_bl[value_col])
            if verbose == 2:
                print('Baseline reset to last visit before the relapse and out of its influence%s (%s, %s=%.1f)'
                      %(': visit #' + str(bl_idx + 1) if bl_idx is not None else '',
                        (global_start + datetime.timedelta(days=bl[date_col].item())).date(),
                        outcome, bl[value_col]))

            # Identify the first possible event
            end = nvisits - 1 if np.isnan(relapse_id.loc[irel, 'closest_vis+']) else relapse_id.loc[irel, 'closest_vis+']
            change_idx = next((n for n in range(bl_idx + 1, end + 1)
                               if within_influence(rel=relapse_id.loc[irel, rdate_col], vis=data_id.loc[n, date_col],
                                                     dist0=relapse_assoc[0], dist1=relapse_assoc[1])),
                              None)
            #change_idx = relapse_id.loc[irel, 'closest_vis+']

            # Look at *all* visits within `relapse_assoc` days from relapse and identify first CONFIRMED change (if any)
            confirmed = False
            ch_idx_tmp = change_idx
            while (not confirmed and change_idx is not None and ch_idx_tmp < nvisits
                and (#data_id.loc[ch_idx_tmp,date_col] - relapse_id.loc[irel,rdate_col] <= relapse_assoc[0]
                    within_influence(rel=relapse_id.loc[irel, rdate_col], vis=data_id.loc[ch_idx_tmp, date_col],
                                     dist0=relapse_assoc[0], dist1=relapse_assoc[1])
                   )):

                # Look at *all* visits within `relapse_assoc` days from relapse and identify first change (if any)
                stable = True
                while (stable and change_idx is not None and ch_idx_tmp < nvisits
                    and (#data_id.loc[ch_idx_tmp, date_col] - relapse_id.loc[irel, rdate_col] <= relapse_assoc[0]
                        within_influence(rel=relapse_id.loc[irel, rdate_col], vis=data_id.loc[ch_idx_tmp, date_col],
                                         dist0=relapse_assoc[0], dist1=relapse_assoc[1])
                       )):
                    change_idx = ch_idx_tmp
                    stable = not isevent_loc(data_id.loc[change_idx, value_col], bl[value_col], type='wors') # no worsening
                    ch_idx_tmp = change_idx + 1

                if verbose == 2 and not stable:
                    print('%s change found: visit #%d (%s, %s=%.1f)' %(outcome, change_idx + 1,
                                (global_start + datetime.timedelta(
                                    days=data_id.loc[change_idx, date_col].item())).date(),
                                outcome, data_id.loc[change_idx, value_col]))

                if (change_idx is None  # no change
                    or not within_influence(rel=relapse_id.loc[irel, rdate_col],
                        vis=data_id.loc[change_idx, date_col],
                        dist0=relapse_assoc[0], dist1=relapse_assoc[1])  # change is out of relapse influence
                    or not isevent_loc(data_id.loc[change_idx, value_col], bl[value_col], type='wors')  # no worsening
                    ):
                    if verbose == 2:
                            print('No relapse-associated worsening')
                    confirmed = False
                    ch_idx_tmp = change_idx + 1
                    continue
                elif (change_idx == nvisits - 1
                      and data_id.loc[change_idx, date_col] - data_id.loc[0, date_col] <= impute_max_fu
                      and np.random.binomial(1, impute_last_visit, 1)):
                    confirmed = True
                    conf_idx = nvisits - 1
                else:
                    change_idx = int(change_idx)
                    conf_idx = [[x for x in range(change_idx + 1, nvisits)
                        if c[0] <= data_id.loc[x, date_col] - data_id.loc[change_idx, date_col] <= c[1] # date in confirmation range
                                 and data_id.loc[x, 'closest_rel-'] >= relapse_to_conf[0]  # occurring out of influence of last relapse
                                 and data_id.loc[x, 'closest_rel+'] >= relapse_to_conf[1]  # occurring out of influence of next relapse
                                 and data_id.loc[x, validconf_col]]  # can be used as confirmation
                        for c in conf_window]
                    conf_idx = [x for i in range(len(conf_idx)) for x in conf_idx[i]]
                    conf_idx = None if len(conf_idx)==0 else min(conf_idx)  # first available confirmation visit

                    confirmed = (conf_idx is not None  # confirmation visits available
                        and all([isevent_loc(data_id.loc[x, value_col], bl[value_col], type='wors')
                             for x in range(change_idx + 1, conf_idx + 1)]))  # increase is confirmed over the confirmation window

                # CONFIRMED PROGRESSION:
                # ---------------------
                if confirmed:
                    last_conf = data_id.loc[conf_idx, date_col]
                    valid_prog = 1
                    if sust_raw_days:
                        next_nonsust = next((x for x in range(conf_idx + 1, nvisits) # next value found
                        if not isevent_loc(data_id.loc[x, value_col], bl[value_col], type='wors')  # worsening not sustained
                                        ), None)
                        valid_prog = (next_nonsust is None) or (data_id.loc[next_nonsust-1, date_col]
                                        - data_id.loc[change_idx, date_col]) > sust_raw_days
                    if valid_prog:
                        sust_idx = next((x - 1 for x in range(conf_idx + 1, nvisits)  # last visit within predetermined "sustained" interval
                                    if (data_id.loc[x, date_col] - data_id.loc[change_idx, date_col]) > sust_raw_days
                                        ), None)
                        sust_idx = nvisits - 1 if sust_idx is None else sust_idx
                        end_idx = sust_idx if irel == len(relapse_id) - 1 else min(
                                    max(relapse_id.loc[irel + 1, 'closest_vis-'], conf_idx),  #relapse_id.loc[irel + 1, 'closest_vis+'] - 1
                                            sust_idx)
                        # Set value change as the minimum within the predetermined "sustained" interval (before the following relapse):
                        value_change = max(data_id.loc[change_idx:end_idx, value_col].min() - bl[value_col], 0)  # NB: PANDAS SLICING WITH .loc INCLUDES THE RIGHT END!!
                        # Detect potential bumps:
                        bump = data_id[value_col] - data_id.loc[change_idx:end_idx, value_col].min()  # values exceeding the minimum
                        bump = np.maximum(bump.loc[change_idx:conf_idx], 0)
                        data_id.loc[change_idx:conf_idx, bump_col] = data_id.loc[change_idx:conf_idx, bump_col] + bump

                        # Store info
                        delta_raw.append(value_change)
                        raw_dates.append(data_id.loc[change_idx, date_col])
                        if include_raw_dates:
                            raw_events.append([subjid, global_start + datetime.timedelta(
                                    days=data_id.loc[change_idx, date_col].item())]) #_d_# data_id.loc[change_idx,date_col]

                        if verbose == 2:
                            print('Confirmed%s RAW on visit #%d (%s); delta = %.1f, max bump delta = %.1f'
                                  %(' sustained' if sust_raw_days > 0 else '', change_idx + 1,
                                    (global_start + datetime.timedelta(
                                    days=data_id.loc[change_idx, date_col].item())).date(),
                                  value_change, bump.max()))
                    else:
                        end_idx = next_nonsust - 1
                        bump = data_id[value_col] - data_id.loc[next_nonsust, value_col]
                        bump = np.maximum(bump.loc[change_idx:end_idx], 0)
                            # NB: PANDAS SLICING WITH .loc INCLUDES THE RIGHT END!
                        data_id.loc[change_idx:end_idx, bump_col] = data_id.loc[change_idx:end_idx, bump_col] + bump
                        if verbose == 2:
                            print('Change (visit #%d, %s) confirmed but not sustained for >=%s days\n-- only subtracting the \"bump\" (max delta = %.1f)'
                                  %(change_idx + 1, (global_start + datetime.timedelta(
                                    days=data_id.loc[change_idx, date_col].item())).date(),
                                    sust_raw_days, bump.max()))

                # NO confirmation:
                # ----------------
                else:
                    if verbose == 2:
                        print('Change not confirmed')
                    if conf_idx is not None:
                        next_nonconf = next((x for x in range(change_idx + 1, conf_idx + 1)
                                         if not isevent_loc(data_id.loc[x, value_col], bl[value_col], type='wors')
                             ))
                        # end_idx = next_nonconf - 1 if irel == len(relapse_id) - 1 else min(relapse_id.loc[irel + 1, 'closest_vis-'], next_nonconf - 1)
                        end_idx = next_nonconf - 1
                        bump = data_id[value_col] - data_id.loc[next_nonconf, value_col]
                    else:
                        end_idx = change_idx
                        bump = data_id[value_col] - bl[value_col]
                    bump = np.maximum(bump.loc[change_idx:end_idx], 0)
                    data_id.loc[change_idx:end_idx, bump_col] = data_id.loc[change_idx:end_idx, bump_col] + bump
                    # NB: PANDAS SLICING WITH .loc INCLUDES THE RIGHT END!!
                    last_conf = end_idx
                    if verbose == 2:
                        print('-- only subtracting the \"bump\"  (max delta = %.1f)'
                              %bump.max())
                    ch_idx_tmp = max(end_idx + 1, change_idx + 1)

        if verbose == 2:
            print('Examined all relapses: end process')

        for d_value, date in zip(delta_raw, raw_dates):
            data_id.loc[data_id[date_col]>=date, ra_col]\
                = data_id.loc[data_id[date_col]>=date, ra_col] + d_value

        data_id[ri_col] = np.maximum(data_id[value_col] - data_id[ra_col] - data_id[bump_col], 0)
        if len(data_id)>0:
            data_id[ri_col] = data_id[ri_col] - global_bl[value_col] #data_id.loc[glob_bl_idx,ri_col]
            if subtract_bl:
                data_id[value_col+'-bl'] = data_id[value_col] - global_bl[value_col] #data_id.loc[glob_bl_idx,value_col]
        if include_bl:
            data_id['bl_'+value_col] = global_bl[value_col]

        if include_rel_num and len(data_id)>0:
            for date in relapse_dates:
                data_id.loc[data_id[date_col] >= date, 'relapse_num'
                   ] = data_id.loc[data_id[date_col] >= date, 'relapse_num'] + 1

        # Remove rows of dropped visits
        ind = data_sep.index[np.where(data_sep[subj_col]==subjid)[0]]
        ind = ind[:-len(data_id)] if len(data_id)>0 else ind
        data_sep = data_sep.drop(index=ind)

        # Update collective dataframe
        if len(data_id)>0:
            data_sep.loc[data[subj_col]==subjid,:] = data_id.drop(columns=['closest_rel-', 'closest_rel+']).values

    data_sep[date_col] = [global_start + datetime.timedelta(
                    days=int(data_sep.loc[ii,date_col])) for ii in data_sep.index]
    data_sep[date_col] = pd.to_datetime(data_sep[date_col])

    return (data_sep, pd.DataFrame(raw_events, columns=[subj_col, date_col])) if include_raw_dates else data_sep


#####################################################################################
