from function import func
import random
import time
import math

random.seed(2021)

#def func(x,y):
#  return x*y

def timer(func):
    def wrapper( *args , **kwargs ):
        start = time.perf_counter()
        v = func( *args , **kwargs )
        end = time.perf_counter()
        print(f"{func.__name__} run {end-start}s")
        return v
    return wrapper

# create xy pair
def make_pair( x_range:list , y_range:list):
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    return [  func(x,y) ,  x , y ]


@timer
def beam_search(  x_range:list , y_range:list , search_range = 2, successors = 2 , k = 100 , steps=1000 ):
    
    min_ans = [float('inf'), 0 , 0  ]

    lin_x =  list(linspace( *x_range , num =  k ) )
    lin_y = list(linspace( *y_range , num =   k  ))

    search=  [  [func(x_,y_) ,x_,y_]  for x_ in lin_x for y_ in lin_y  ]


    break_times = 20

    # successors
    for s in range(steps):  

        search =  sorted(search , key=lambda x:x[0])
        search = search[:successors]
        
        if break_times==0 or abs (search[0][0] - min_ans[0])<0.001:
            break
  
        if search[0][0] < min_ans[0] :
            min_ans = search[0]
        else:
            break_times-=1
            
        new_search  = []

        for p in search:

            low_x_range = [  max(  x_range[0] ,  p[1]-search_range ) , p[1]  ]
            low_y_range = [  max(  y_range[0] ,  p[2]-search_range ) , p[2]  ]   
            high_x_range = [  p[1]   , min(  x_range[1] ,  p[1] + search_range )   ]
            high_y_range = [  p[2]   , min(  y_range[1] ,  p[2] + search_range )   ]

            for k_ in range(k//4):
                #new_search.append(make_pair(new_x_range,new_y_range))
                new_search.append(make_pair(low_x_range,low_y_range))
                new_search.append(make_pair(low_x_range,high_y_range))
                new_search.append(make_pair(high_x_range,low_y_range))
                new_search.append(make_pair(high_x_range,high_y_range))

        while len(new_search) < k:
          new_search.append(  make_pair(x_range,y_range)  )
        
        search = new_search

    return min_ans


def linspace(start, stop, num=50, endpoint=False):
    num = int(num)
    start = start * 1.
    stop = stop * 1.

    if num == 1:
        yield stop
        return
    if endpoint:
        step = (stop - start) / (num - 1)
    else:
        step = (stop - start) / num

    for i in range(num):
        yield start + step * i


@timer
def brute_force(  x_range:list , y_range:list , steps = 10):
    
    min_v = float('inf')   
    min_ans = None

    x =  linspace( *x_range , num = steps)
    y = list(linspace( *y_range , num = steps))
    
    for x_ in x:
        for y_ in y:
            v = func(x_,y_)
            if v < min_v:
                min_v = v
                min_ans = [v,x_,y_]
    
    return min_ans

def main():
    
    # read x and y range
    with open('./input.txt' ,'r') as f:
        x_range = [ float(i) for i in f.readline().replace('\n','').split(',')]
        y_range = [ float(i) for i in  f.readline().replace('\n','').split(',')]
    
    # brute search 
    brute_search_result = brute_force(x_range,y_range,steps=1000)
    print(f"{  brute_search_result[1]  }\n{ brute_search_result[2] }\n{ round(brute_search_result[0],3) }" )

    # beam search
    k = int(  max( (x_range[1]-x_range[0]) , ( y_range[1]-y_range[0])) ) *  2
    k = 1000
    steps = 100
    successors = int(k/10)
    search_range = 1
    beam_search_result = beam_search(x_range,y_range,search_range =  search_range ,successors = successors , k = k , steps=steps)
    print(f"{  beam_search_result[1]  }\n{beam_search_result[2] }\n{ round(beam_search_result[0],3)}" )
    

if __name__ == "__main__":
    main()
    
