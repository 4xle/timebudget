import pytest

# stupidly simple integration tests.

def test_demo1():
    # Note this doesn't properly test the atexit.register  Oh well.
    import demo1

def test_demo2():
    import demo2

def test_demo3():
    import demo3

def test_demo4():
    import demo4

def test_demo5():
    print("testing uniform units=False")
    import demo5

def test_demo6():
    print("testing uniform units=True")
    import demo6

def test_demo7():
    print("testing simple recursion")
    import demo7

def test_demo8():
    print("testing complex recursion")
    import demo8
