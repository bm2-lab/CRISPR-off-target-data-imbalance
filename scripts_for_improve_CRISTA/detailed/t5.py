

global y

def foo(i):
    print(f'{i} / {y}')


y = 10
list(map(foo, range(5)))
y = 20
list(map(foo, range(10)))