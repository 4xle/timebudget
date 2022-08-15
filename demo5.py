import time
from timebudget import timebudget
timebudget.set_units("microsecond",uniform_units=False)
timebudget.report_at_exit()

@timebudget
def possibly_slow():
    print("slow")
    time.sleep(0.6)
    

@timebudget
def should_be_fast():
    print("quick")
    time.sleep(0.3)

@timebudget
def should_be_different():
    print("long")
    time.sleep(1)


possibly_slow()
possibly_slow()
should_be_fast()
should_be_fast()
possibly_slow()
should_be_different()
