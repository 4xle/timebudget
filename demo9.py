import time
from timebudget import timebudget
timebudget.set_units("microsecond",uniform_units=True)
timebudget.report_at_exit()



@timebudget.cached
def simple_recursion_test(countdown=5):
    time.sleep(0.1)
    if countdown == 0:
        return None
    else:
        return simple_recursion_test(countdown-1)


@timebudget
def double_recursion_test(countdown=1):
    if countdown == 0:
        return None
    else:
        return double_recursion_test(countdown-1)


@timebudget
def outer_recursion(countdown=1):
    simple_recursion_test(2)

    simple_recursion_test(1)

    double_recursion_test(1)




# @timebudget
# def possibly_slow():
#     print('slow', end=' ', flush=True)
#     time.sleep(0.06)

# @timebudget
# def should_be_fast():
#     print('quick', end=' ', flush=True)
#     time.sleep(0.03)

# @timebudget
# def outer_loop():
#     possibly_slow()
#     possibly_slow()
#     should_be_fast()
#     should_be_fast()
#     possibly_slow()
#     time.sleep(0.2)
#     print("dance!")

# for n in range(7):
#     outer_loop()
outer_recursion(2)

timebudget.report('outer_recursion')
