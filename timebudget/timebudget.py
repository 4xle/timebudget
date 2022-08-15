"""Timebudget is a stupidly-simple tool to help measure speed. 
Two main ways to use:

1) Put code to measure in a with statement:

with timebudget("load file"):
    text = open(filename,'rt').readlines()

2) Annotate any function to measure how long it takes

from timebudget import timebudget

@timebudget  # times how long we spend in this function
def my_possibly_slow_function(*args):
    # Do something

By default it prints time measurements immediately. Or:
timebudget.set_quiet()  # Sets `quiet=True` as global default
timebudget.report()  # prints a summary of all annotated functions
"""
import atexit
from collections import defaultdict
from functools import wraps
import sys
import time
from typing import Callable, Optional, Union
import warnings
import pint
import pandas as pd
from numpy import nan,ptp
import pint_pandas
pint_pandas.PintType.ureg.default_format = "~P"
from rich_dataframe import prettify
from pprint import pprint


__all__ = [
    'timebudget', 
    'annotate', 
    'report', 
    'set_quiet',
]

class TimeBudgetRecorder():
    """The object that stores times used for different things and generates reports.
    This is mostly used through annotation and with-block helpers.
    """

    def __init__(self, quiet_mode:bool=False,units='microsecond',uniform_units=False):
        self.quiet_mode = quiet_mode
        self.reset()
        self.out_stream = sys.stdout
        self.ureg = pint.UnitRegistry()
        pint.set_application_registry(self.ureg)
        self.ureg.define('cycle = 1 * turn = cyc')
        self.ureg.define('fraction = [] = frac')
        self.ureg.define('percent = 1e-2 frac = pct')
        self.ureg.define('ppm = 1e-6 fraction')
        self.timeunit = self.ureg[units]
        self.uniform_units = uniform_units
        if self.uniform_units:
            self.ureg.default_format='~'
            pint_pandas.PintType.ureg.default_format = "~"
        else:
            self.ureg.default_format='~#P'
            pint_pandas.PintType.ureg.default_format = "~#P"


    def reset(self):
        """Clear all stats collected so far.
        """
        self.start_times = {}
        # self.elapsed_total = defaultdict(float)  # float defaults to 0
        self.elapsed_total = {}  # float defaults to 0
        # self.elapsed_min = defaultdict(float)  # float defaults to 0
        # self.elapsed_max = defaultdict(float)  # float defaults to 0
        # self.elapsed_cnt = defaultdict(int)  # int defaults to 0

    def _print(self, msg:str):
        self.out_stream.write(msg)
        self.out_stream.write("\n")
        self.out_stream.flush()

    def start(self, block_name:str):
        if block_name in self.start_times:
            # End should clear out the record, so something odd has happened here.
            # try/finally should prevent this, but sometimes it doesn't.
            warnings.warn(f"timebudget is confused: timebudget.start({block_name}) without end")
        self.start_times[block_name] = time.monotonic_ns()
        # print(f"{self.start_times[block_name]=}")

    def end(self, block_name:str, quiet:Optional[bool]=None) -> float:
        """Returns number of ms spent in this block this time.
        """
        if quiet is None:
            quiet = self.quiet_mode
        if block_name not in self.start_times:
            warnings.warn(f"timebudget is confused: timebudget.end({block_name}) without start")
            return float('NaN')
        elapsed = (time.monotonic_ns() - self.start_times[block_name])
        # print(f"{elapsed=}")
        if block_name not in self.elapsed_total:
            self.elapsed_total[block_name] = [elapsed]
        else:
            self.elapsed_total[block_name].append(elapsed)
        # self.elapsed_cnt[block_name] += 1
        del self.start_times[block_name]
        if not quiet:
            self._print(f"{block_name} took {(elapsed * self.ureg.nanosecond).to(self.timeunit)}")
            # pass
        return elapsed

    # def time_format(self, ns_duration:int) -> str:
    #     assert ns_duration >= 0
    #     value = ns_duration * self.ureg.nanosecond
    #     return value.to_compact()


    def _formatResults(self, res):
        if self.uniform_units:
            formattedDict = {
                'name': f"{res['name']:>25s}",
                'total': f"{res['total']:8.3f~}",
                'cnt': f"{res['cnt']}",
                # 'pct':f"{pct: 6.1f}",
                # 'avg_cnt':f"{avg_cnt:8.3f}",
                'avg': f"{res['avg']:8.3f~}",
                'min': f"{res['min']:8.3f~}",
                'max': f"{res['max']:8.3f~}",
                'diff': f"{res['diff']:8.3f~}",
                'sd':f"{res['sd']:8.3f~}",
                'var':f"{res['var']:8.3f~P}"
            }
        else:
            formattedDict = {
                'name': f"{res['name']:>25s}",
                'total': f"{res['total']:8.3f~#P}",
                'cnt': f"{res['cnt']}",
                # 'pct':f"{pct: 6.1f}",
                # 'avg_cnt':f"{avg_cnt:8.3f}",
                'avg': f"{res['avg']:8.3f~#P}",
                'min': f"{res['min']:8.3f~#P}",
                'max': f"{res['max']:8.3f~#P}",
                'diff': f"{res['diff']:8.3f~#P}",
                'sd':f"{res['sd']:8.3f~#P}",
                'var':f"{res['var']:8.3f~#P}"
            }

        return formattedDict


    def _findSmallestPintUnit(self, quantity):
        # print(quantity)
        print(quantity.to_compact().units)
        return quantity.to_compact().units



    def _compileResults(self):


        internalData= {k:pd.Series(v,dtype=f"pint[nanosecond]") for k,v in self.elapsed_total.items()}

        internalDataFrame = pd.DataFrame(internalData).T
        originalDtypes = internalDataFrame.dtypes
        # print(originalDtypes)
        internalDataFrame = internalDataFrame.pint.dequantify()

        numberOfRows = len(internalDataFrame.index)
        rangeForDataRows = range(0,numberOfRows)

        timeFields = ["avg","min","max","range","sum","sd","var"]
        counterFields = ["calls"]
        aggregateFields = ["pct"]
        
        reportDataFrame = pd.DataFrame(index=[k for k in self.elapsed_total])

        # add the fields first in the desired order so that logical ops afterwards can be organized without impacting the table order
        fieldsToAdd = timeFields + counterFields + aggregateFields
        for f in fieldsToAdd:
            reportDataFrame[f] = 0
        # print(reportDataFrame)
        
        reportDataFrame['calls'] = pint_pandas.PintArray(internalDataFrame.count(axis=1),dtype=self.ureg.cycle)
        reportDataFrame['sum'] = pint_pandas.PintArray(internalDataFrame.sum(axis=1,skipna=True),dtype=f"pint[nanosecond]")
        reportDataFrame['avg'] = pint_pandas.PintArray(internalDataFrame.mean(axis=1,skipna=True).round(2),dtype=f"pint[nanosecond]")
        
        reportDataFrame['min'] = pint_pandas.PintArray(internalDataFrame.min(axis=1,skipna=True),dtype=f"pint[nanosecond]")
        reportDataFrame['max'] = pint_pandas.PintArray(internalDataFrame.max(axis=1,skipna=True),dtype=f"pint[nanosecond]")
        reportDataFrame['range'] = reportDataFrame['max'] - reportDataFrame['min']
        
        reportDataFrame['sd'] = pint_pandas.PintArray(internalDataFrame.std(axis=1,skipna=True).round(2),dtype=f"pint[nanosecond]")
        reportDataFrame['var'] = pint_pandas.PintArray(internalDataFrame.var(axis=1,skipna=True).round(2),dtype=f"pint[nanosecond]")
        longestRuntime = reportDataFrame['sum'].max()
        reportDataFrame['pct'] = (reportDataFrame['sum'] / longestRuntime) * 100 * self.ureg.pct


        if self.uniform_units:
            for f in timeFields:
                reportDataFrame[f] = reportDataFrame[f].pint.to(self.timeunit)
        else:
            for f in timeFields:
                reportDataFrame[f] = reportDataFrame[f].pint.to(self._findSmallestPintUnit(reportDataFrame[f].min()))


            


        # reportDataFrame['pct'] = reportDataFrame['pct'].round(2)
        # reportDataFrame = reportDataFrame.round(2)
        # print(reportDataFrame.round(2))
        # print(reportDataFrame)
        # print(reportDataFrame.dtypes)
        reportDataFrame = reportDataFrame.sort_values("pct",ascending=False)
        reportDataFrame = reportDataFrame.pint.dequantify().round(2)
        # exit()


        return reportDataFrame


    def report(self, percent_of:str=None, reset:bool=False):
        """Prints a report summarizing all the times recorded by timebudget.
        If percent_of is specified, then times are shown as a percent of that function.
        If `reset` is set, then all stats will be cleared after this report.
        If `uniform_units` is set, then all time values will use the same (smallest) unit value.
        """

        results = self._compileResults()

        # pprint(self.elapsed_total)
        # exit()


        # df = pd.DataFrame.from_dict(results)
        # print(df)
        # print(df.dtypes)
        # df_ = df.pint.quantify(level=-1)
        # pprint(df_)
        # # df.pint.dequantify()

        print(prettify(results,delay_time=0.1))
        # exit()
        

        # if percent_of:
        #     # assert percent_of in self.elapsed_cnt, f"Can't generate report for unrecognized block {percent_of}"
        #     self._print(f"timebudget report per {percent_of} cycle...")
        #     total_elapsed = sum(self.elapsed_total[percent_of])
        #     # total_cnt = self.elapsed_cnt[percent_of]
        #     total_cnt = len(self.elapsed_total[percent_of]) * self.ureg.cycle
        #     for res in results:
        #         formattedDict = self._formatResults(res)
        #         formattedDict['avg'] = res['total'] / total_cnt
        #         formattedDict['pct'] = 100.0 * res['total'] / total_elapsed
        #         formattedDict['avg_cnt'] = f"{res['cnt'] / total_cnt:8.1f~P}"




        #         self._print(f"{formattedDict['name']}:{formattedDict['pct']}% avg, sd {formattedDict['sd']}, var {formattedDict['var']}, min {formattedDict['min']}, max {formattedDict['max']}, range {formattedDict['diff']}, {formattedDict['avg_cnt']} calls/cycle, total time:{formattedDict['total']}")


        #         # if self.uniform_units:
        #         #     self._print(f"{res['name']:>25s}:{pct: 6.1f}% avg {avg:8.3f},sd {res['sd']:8.3f}, var {res['var']:8.3f}, range {res['diff']:8.3f} @{avg_cnt:8.3f} calls/cycle ")
        #         # else:
        #         #     self._print(f"{res['name']:>25s}:{pct: 6.1f~#P}% avg {avg:8.3f~#P},sd {res['sd']:8.3f~#P}, var {res['var']:8.3f~#P}, range {res['diff']:8.3f~#P} @{avg_cnt:8.3f~#P} calls/cycle ")
        # else:
        #     self._print("timebudget report...")
        #     for res in results:

        #         # print(res)
        #         # diff = res['max'] - res['min']
        #         # sd = (res['avg'].m**0.5) * self.ureg.ns

        #         formattedDict = self._formatResults(res)
        #         # print(formattedDict)

        #         self._print(f"{formattedDict['name']}:{formattedDict['avg']} avg, sd {formattedDict['sd']}, var {formattedDict['var']}, min {formattedDict['min']}, max {formattedDict['max']}, range {formattedDict['diff']}, {formattedDict['cnt']} calls, total time:{formattedDict['total']}")


        #         # if self.uniform_units:
        #         #     self._print(f"{res['name']:>25}:{res['avg']:8.3f~P} avg,sd {res['sd']:8.3f~P}, var {res['var']:8.3f~P}, range {res['diff']:8.3f~P} for {res['cnt']: 6d~P} calls")
        #         # else:
        #         #     self._print(f"{res['name']:>25}:{res['avg']:8.3f~#P} avg,sd {res['sd']:8.3f~#P}, var {res['var']:8.3f~#P}, range {res['diff']:8.3f~#P} for {res['cnt']: 6d} calls")
        if reset:
            self.reset()

        


_default_recorder = TimeBudgetRecorder()  


def annotate(func:Callable, quiet:Optional[bool]):
    """Annotates a function or code-block to record how long the execution takes.
    Print summary with timebudget.report
    """
    name = func.__name__
    @wraps(func)
    def inner(*args, **kwargs):
        _default_recorder.start(name)
        try:
            return func(*args, **kwargs)
        finally:
            _default_recorder.end(name, quiet)
    return inner

class _timeblock():
    """Surround a code-block with a timer as in
        with timebudget('loadfile'):
    """

    def __init__(self, name:str, quiet:Optional[bool]):
        self.name = name
        self.quiet = quiet

    def __enter__(self):
        _default_recorder.start(self.name)

    def __exit__(self, typ, val, trace):
        _default_recorder.end(self.name, self.quiet)


def annotate_or_with_block(func_or_name:Union[Callable, str], quiet:Optional[bool]=None):
    if callable(func_or_name):
        return annotate(func_or_name, quiet)
    if isinstance(func_or_name, str):
        return _timeblock(func_or_name, quiet)
    raise RuntimeError("timebudget: Don't know what to do. Either @annotate or with:block")


def set_quiet(quiet:bool=True):
    """Tell timebudget not to print time measurements on every call, but instead
    to save them only for a report.  
    Alternately, you can reverse this by calling set_quiet(False).
    """
    _default_recorder.quiet_mode = quiet

def set_units(units='millisecond',uniform_units=True):
    _default_recorder.timeunit = _default_recorder.ureg[units]
    _default_recorder.uniform_units = uniform_units


# Create shortcuts for export
timebudget = annotate_or_with_block
report = _default_recorder.report
timebudget.report = report
timebudget.__doc__ = __doc__
timebudget.set_quiet = set_quiet
timebudget._default_recorder = _default_recorder
timebudget.set_units = set_units

def report_at_exit(block_name:str=None, uniform_units=True):
    if block_name:
        atexit.register(lambda: report(block_name, uniform_units))
    else:
        atexit.register(report)


timebudget.report_at_exit = report_at_exit
timebudget.report_atexit = report_at_exit  # for backwards compat with v0.6

