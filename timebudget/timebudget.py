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
from collections import defaultdict,Counter
from functools import wraps, partial
import sys
import time
from typing import Callable, Optional, Union,List
from dataclasses import dataclass,field
import tabulate
import warnings
import pint
import pandas as pd
from tqdm import tqdm
from functools import cache,cached_property
# from shutil import get_terminal_size
# pd.set_option('display.width', get_terminal_size()[0])
# from numpy import nan,ptp
import numpy as np
import pint_pandas
pint_pandas.PintType.ureg.default_format = "~P"
# from rich_dataframe import prettify
from pprint import pprint


__all__ = [
    'timebudget', 
    'annotate', 
    'report', 
    'set_quiet',
]



# this data class exists to store timestamps for function calls as fast as possible 
# duration and other features are computed during the report_at_exit phase unless they are 
# needed earlier
@dataclass
class TimeDurationRecord:
    name:str
    startns:int
    endns:int = -1
    duration:int = -1
    finished:bool = False
    pauses:List = field(default_factory=list)
    resumes:List = field(default_factory=list)

    def __init__(self,name):
        self.startns = time.monotonic_ns()
        self.name = name
        self.pauses = []
        self.resumes = []


    def end(self):
        self.endns = time.monotonic_ns()
        self.finished = True

    def pause(self):
        self.pauses.append(time.monotonic_ns())

    def resume(self):
        self.resumes.append(time.monotonic_ns())


    def computeDuration(self):
        # print(f"{self.startns=},{self.endns=}")
        # print(f"{self.pauses=},{self.resumes=}")
        # print(f"{self.resumes[0]=},{self.pauses[0]=},{self.resumes[0]-self.pauses[0]}")
        # print(f"{self.endns-self.startns=}")
        # print(f"total paused duration={sum([a-b for a,b in zip(self.resumes,self.pauses)])}")
        # print(f"duration around pauses:{(self.endns - self.startns) - sum([a-b for a,b in zip(self.resumes,self.pauses)])}")
        self.duration = self.endns - self.startns - sum([a-b for a,b in zip(self.resumes,self.pauses)])
        # print(f"{self.duration=}")





class TimeBudgetRecorder():
    """The object that stores times used for different things and generates reports.
    This is mostly used through annotation and with-block helpers.
    """

    def __init__(self, quiet_mode:bool=False,units='microsecond',uniform_units=False,sortbyKey=""):
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
        self.sortbyKey=sortbyKey
        self.cache_data = {}
        if self.uniform_units:
            self.ureg.default_format='~'
            pint_pandas.PintType.ureg.default_format = "~"
        else:
            self.ureg.default_format='~#P'
            pint_pandas.PintType.ureg.default_format = "~#P"

        self.globalStartTime = time.monotonic_ns()


    def reset(self):
        """Clear all stats collected so far.
        """
        self.start_times = defaultdict(list)
        self.last_started_idx = defaultdict(list)
        self.last_finished_idx = defaultdict(list)
        self.actively_running = Counter()
        self.started_count = Counter()
        self.finished_count = Counter()
        # self.elapsed_total = defaultdict(float)  # float defaults to 0
        self.elapsed_total = defaultdict(list)  # float defaults to 0
        # self.elapsed_min = defaultdict(float)  # float defaults to 0
        # self.elapsed_max = defaultdict(float)  # float defaults to 0
        # self.elapsed_cnt = defaultdict(int)  # int defaults to 0

    def _print(self, msg:str):
        self.out_stream.write(msg)
        self.out_stream.write("\n")
        self.out_stream.flush()


    def start(self, block_name:str):
        # print(f"started block={block_name}")

        # some call to the function has been called recursively earlier
        # include a pausing timestamp 
        # if len(self.last_started_idx[block_name])-1 != self.last_started_idx[block_name][-1]:
        if self.actively_running[block_name] > 0: 
            # activeIdx = self.last_started_idx[block_name][-1]
            self.start_times[block_name][self.last_started_idx[block_name][-1]].pause()


        # if len(self.start_times[block_name]) > 0 and self.start_times[block_name][-1].finished is False:
        #     # End should clear out the record, so something odd has happened here.
        #     # try/finally should prevent this, but sometimes it doesn't.
        #     warnings.warn(f"timebudget is confused: timebudget.start({block_name}) without end")

        self.start_times[block_name].append(TimeDurationRecord(block_name))

        # add the index of the started record to the monitoring list
        self.last_started_idx[block_name].append(len(self.start_times[block_name])-1)
        self.actively_running[block_name] += 1
        # print(f"{self.start_times[block_name]=}")






    def end(self, block_name:str, quiet:Optional[bool]=None):
        """Returns number of ms spent in this block this time.
        """
        # if quiet is None:
        #     quiet = self.quiet_mode
        # if block_name not in self.start_times:
        #     warnings.warn(f"timebudget is confused: timebudget.end({block_name}) without start")
        #     return float('NaN')
        # elapsed = (time.monotonic_ns() - self.start_times[block_name])

        # print(f"{self.last_started_idx=}")
        # print(f"{self.started_count=}")
        # print(f"{self.actively_running=}")

        # activeIdx = self.last_started_idx[block_name].pop()

        self.start_times[block_name][self.last_started_idx[block_name].pop()].end()
        self.actively_running[block_name] -=1
        # self.finished_count[block_name] += 1

        # if there are still indicies for actively running functions of the same name

        if self.actively_running[block_name] > 0:
            # activeIdx = self.last_started_idx[block_name][-1]
            self.start_times[block_name][self.last_started_idx[block_name][-1]].resume()


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
        # print(quantity.to_compact().units)
        return quantity.to_compact().units


    def roundFieldAfterConversion(self,field,conversionString):
        field = field.pint.to(conversionString)
        field = pint_pandas.PintArray(np.around(field.values.quantity.m,2),dtype=f"pint[{conversionString}]")
        return field



    def _compileResults(self):
        # results are default nanosecond, not typed with pint until the very end to reduce overhead on conversions
        internalData= {k:pd.Series(v) for k,v in self.elapsed_total.items()}

        internalDataFrame = pd.DataFrame(internalData)

        # print(tabulate.tabulate(internalDataFrame, tablefmt="github", headers="keys", showindex="always"))
        # exit()
        # print(internalDataFrame.sum(skipna=True))
        originalDtypes = internalDataFrame.dtypes
        # print(originalDtypes)
        # internalDataFrame = internalDataFrame.pint.dequantify()

        numberOfRows = len(internalDataFrame.index)
        rangeForDataRows = range(0,numberOfRows)

        fieldUnitMapping = {
            "calls":"dimensionless",
            "pct":"dimensionless",
            "avg":"nanosecond/cycle",
            "sum":"nanosecond",
            "min":"nanosecond",
            "max":"nanosecond",
            "range":"nanosecond",
            "sd1":"nanosecond",
            "sd2":"nanosecond",
            "cov1":"dimensionless",
        }

        conversionUnitMapping = {
        "calls":"dimensionless",
            "pct":"dimensionless",
            "avg":f"{self.timeunit}/cycle",
            "sum":f"{self.timeunit}",
            "min":f"{self.timeunit}",
            "max":f"{self.timeunit}",
            "range":f"{self.timeunit}",
            "sd1":f"{self.timeunit}",
            "sd2":f"{self.timeunit}",
            "cov1":"dimensionless"
        }

        def getDomainRange(data):
            return data.max() - data.min()


        def computePct(data):
            return data.sum() / self.totalRunningTime * 100

        def computeStd1(data):
            return data.std(skipna=True)

        def computeStd2(data):
            return data.std(skipna=True)*2

        def computeCov1(data):
            return data.std(skipna=True) / data.mean(skipna=True) 


        operationMapping = {
            "calls":pd.DataFrame.count,
            "pct":computePct,
            "avg":pd.DataFrame.mean,
            "sum":pd.DataFrame.sum,
            "min":pd.DataFrame.min,
            "max":pd.DataFrame.max,
            "range":getDomainRange,
            "sd1":computeStd1,
            "sd2":computeStd2,
            "cov1":computeCov1,
            # "pct":,
            # "sd1max":"nanosecond",
            # "sd2max":"nanosecond",
        }

        # counterFields1 = ["calls"]
        # timeFields = ["avg","sum","min","max","range","sd1","sd2","sd1max","sd2max"] # "var"
        # counterFields2 = ["cov1"]
        # aggregateFields = ["pct"]
        
        rawDataFrame = pd.DataFrame(index=[k for k in self.elapsed_total])
        reportDataFrame = pd.DataFrame(index=[k for k in self.elapsed_total])

        # add the fields first in the desired order so that logical ops afterwards can be organized without impacting the table order
        # fieldsToAdd = counterFields1 + aggregateFields + timeFields + counterFields2
        for f in fieldUnitMapping:
            # rawDataFrame[f] = pint_pandas.PintArray(internalDataFrame.assign(f=lambda x: operationMapping[f](x)),dtype=f"pint[{fieldUnitMapping[f]}]")
            rawDataFrame[f] = pint_pandas.PintArray(np.around(operationMapping[f](internalDataFrame),2),dtype=f"pint[{fieldUnitMapping[f]}]")


        # print(rawDataFrame)
        rawDataFrame['calls'] = rawDataFrame['calls'].astype(int).astype(str) 
        rawDataFrame['pct'] = rawDataFrame['pct'].astype(str) 
        for f,data in self.cache_data.items():
            # print(rawDataFrame['calls'])
            # print(rawDataFrame['calls'][f])
            
            cacheHitPercent = round((data.hits/int(rawDataFrame['calls'][f]))*float(rawDataFrame['pct'][f]),1)
            cacheMissPercent = round((data.misses/int(rawDataFrame['calls'][f]))*float(rawDataFrame['pct'][f]),1)
            rawDataFrame['pct'][f] = f"{rawDataFrame['pct'][f]} ({cacheHitPercent}/{cacheMissPercent}))"

            rawDataFrame['calls'][f] = f"{rawDataFrame['calls'][f]} ({data.hits}/{data.misses}/{data.currsize})"

        if len(self.sortbyKey) == 0:
            rawDataFrame = rawDataFrame.sort_values("pct",ascending=False)
        else:
            rawDataFrame = rawDataFrame.sort_values(self.sortbyKey,ascending=False)
            # print(rawDataFrame)

        for f in fieldUnitMapping:
            if fieldUnitMapping[f] != "dimensionless":
                rawDataFrame[f] = self.roundFieldAfterConversion(rawDataFrame[f], conversionUnitMapping[f])

        if not self.uniform_units:
        #     for f in timeFields:
        #         reportDataFrame[f] = reportDataFrame[f].pint.to(self.timeunit)
        # else:
            for f,v in conversionUnitMapping.items():
                if v != "dimensionless":
                # rawDataFrame[f] = rawDataFrame[f].pint.to(self._findSmallestPintUnit(rawDataFrame[f].min()))
                    rawDataFrame[f] = self.roundFieldAfterConversion(rawDataFrame[f], self._findSmallestPintUnit(rawDataFrame[f].min()))
                # reportDataFrame[f] = reportDataFrame[f].pint.dequantify()
            rawDataFrame = rawDataFrame.pint.dequantify()



        # # if self.uniform_units is False:
        # reportDataFrame = reportDataFrame.pint.dequantify().round(2)
        # # exit()


        return rawDataFrame


    def processTimeRecords(self):
        for k,data in tqdm(self.start_times.items(),desc="Processing time records"):
            for record in data:
                record.computeDuration()
                self.elapsed_total[k].append(record.duration)

        self.start_times = defaultdict(list)

        # print(f"{self.elapsed_total=}")

    def log_cache_data(self, name, data):
        self.cache_data[name] = data


    def report(self, percent_of:str=None, reset:bool=False):
        self.globalEndTime = time.monotonic_ns()
        self.totalRunningTime = self.globalEndTime - self.globalStartTime
        """Prints a report summarizing all the times recorded by timebudget.
        If percent_of is specified, then times are shown as a percent of that function.
        If `reset` is set, then all stats will be cleared after this report.
        If `uniform_units` is set, then all time values will use the same (smallest) unit value.
        """
        self.processTimeRecords()

        results = self._compileResults()

        # print(prettify(results,delay_time=0.1,row_limit=len(self.elapsed_total.keys()),col_limit=len(results.columns)))
        print(tabulate.tabulate(results, tablefmt="github", headers="keys", showindex="always"))
        # print(f"Total logged time:{sum(results['sum'])}")
        print(f"Total running time:{(self.totalRunningTime * self.ureg['nanosecond']).to(self.timeunit)}")
        # exit()
        
        if reset:
            self.reset()

        


_default_recorder = TimeBudgetRecorder()  


def annotate(func:Callable, quiet:Optional[bool],withcache:Optional[bool],cacheproperty:Optional[bool]):
    """Annotates a function or code-block to record how long the execution takes.
    Print summary with timebudget.report
    """
    name = func.__name__

    if withcache:
        if not cacheproperty:
            func = cache(func)
        else:
            func = cached_property(func)
        # print(f"caching added to function:{name}")

    @wraps(func)
    def inner(*args, **kwargs):
        _default_recorder.start(name)
        try:
            return func(*args, **kwargs)
        finally:
            _default_recorder.end(name, quiet)

            if withcache:
                _default_recorder.log_cache_data(name, func.cache_info())

            # if withcache:
            #     print(f"{name=},{func.cache_info()=}")
    return inner


class _timeblock():
    """Surround a code-block with a timer as in
        with timebudget('loadfile'):
    """

    def __init__(self, name:str, quiet:Optional[bool],withcache:Optional[bool],cacheproperty:Optional[bool]):
        self.name = name
        self.quiet = quiet

    def __enter__(self):
        _default_recorder.start(self.name)

    def __exit__(self, typ, val, trace):
        # print(f"Exiting {self.name=}")
        _default_recorder.end(self.name, self.quiet)


def annotate_or_with_block(func_or_name:Union[Callable, str], quiet:Optional[bool]=None,withcache:Optional[bool]=False,cacheproperty:Optional[bool]=False):
    if callable(func_or_name):
        return annotate(func_or_name, quiet,withcache,cacheproperty)
    if isinstance(func_or_name, str):
        return _timeblock(func_or_name, quiet,withcache,cacheproperty)
    raise RuntimeError("timebudget: Don't know what to do. Either @annotate or with:block")



annotate_cache = partial(annotate_or_with_block,quiet=True,withcache=True,cacheproperty=False)
annotate_cached_property = partial(annotate_or_with_block,quiet=True,withcache=True,cacheproperty=True)

# def annotate_or_with_block_cached(func_or_name:Union[Callable, str], quiet:Optional[bool]=None,withcache:Optional[bool]=True,cacheproperty:Optional[bool]=False):
#     if callable(func_or_name):
#         return annotate(func_or_name, quiet,withcache,cacheproperty)
#     if isinstance(func_or_name, str):
#         return _timeblock(func_or_name, quiet,withcache,cacheproperty)
#     raise RuntimeError("timebudget: Don't know what to do. Either @annotate or with:block")

# def annotate_or_with_block_cache_property(func_or_name:Union[Callable, str], quiet:Optional[bool]=None,withcache:Optional[bool]=True,cacheproperty:Optional[bool]=True):
#     if callable(func_or_name):
#         return annotate(func_or_name, quiet,withcache)
#     if isinstance(func_or_name, str):
#         return _timeblock(func_or_name, quiet,withcache)
#     raise RuntimeError("timebudget: Don't know what to do. Either @annotate or with:block")


def set_quiet(quiet:bool=True):
    """Tell timebudget not to print time measurements on every call, but instead
    to save them only for a report.  
    Alternately, you can reverse this by calling set_quiet(False).
    """
    _default_recorder.quiet_mode = quiet


def set_units(units='millisecond',uniform_units=True,sortbyKey=""):
    _default_recorder.timeunit = _default_recorder.ureg[units]
    _default_recorder.uniform_units = uniform_units
    _default_recorder.sortbyKey = sortbyKey


# Create shortcuts for export
timebudget = annotate_or_with_block
timebudget.cache = annotate_cache
timebudget.cached_property = annotate_cached_property
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

