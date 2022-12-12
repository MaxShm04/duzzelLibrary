import math
from itertools import permutations
from math import *
import random
from getpass import getpass
from datetime import datetime



def sum_of_numbers(strt, end):
    return int(((end-strt+1)*(strt+end))/2)

def sum_of_numbers_in_steps(n, start=0, step=1):
    return n*((start-1+(start-1+(n-1)*step))/2)+1

#print(sum_of_numbers_in_steps(3, 5, 8))
#print(sum_of_numbers(-3, 4))

def properDivisors(x):
    ret = []
    for n in range(1, x):
        if x % n == 0:
            ret.append(n)
    # print(ret)
    return ret


def is_even(x):
    if x % 2 == 0:
        return True
    return False


def sumOfNumArr(x):
    c = 0
    for n in x:
        c += n
    # print(sum)
    return c


def BoolArrToSum(x):
    y = 0
    for n in range(0, len(x)):
        if x[n] == False:
            y += n
    return y


def givePeriod(x, y):
    if type(x) == float:
        nk = str(x).split(".")[1]
        vk = str(x).split(".")[0]
        # print(nk)
        period = []
        ind = 0
        zeros = []
        rem = []
        for n in nk:
            if n != "0":
                break
            else:
                zeros.append(n)
        """        
        #print(zeros)
        for n in nk:
            ind +=1
            if n in period:
                #print("again")
                rem = period[:period.index(n)]
                for r in rem:
                    #print(F"remove {r}")
                    period.remove(r)
                if getFollow(period, nk[ind-1:]):
                    #print("finishing")
                    if y == 0:
                        return F'{vk}.{listToString(zeros)}{listToString(rem)}({listToString(period)})'
                    else:
                        return len(period)
                else:
                    period.append(n)
                    period = rem + period
            else:
                #print(F"add {n}")
                period.append(n)
        return x"""
    return False


def getFollow(liste, strin):
    # print(strin)
    ind = 0
    for s in strin:
        ind += 1
        if ind - 1 == len(liste):
            # print(True)
            return True
        elif s != liste[strin.index(s)]:
            # print(False)
            return False


def listToString(x):
    try:
        return "".join(x)
    except:
        outp = ""
        for n in x:
            outp += str(n)
        return outp


def StringToList(x):
    return list(x)


def emptyList(x):
    ret = []
    for n in range(0, x):
        ret.append(0)
    return ret


def resetBoolList(x):
    ret = [False] * len(x)
    return ret


def stringToIntList(x):
    ret = []
    for n in x:
        ret.append(int(n))
    return ret


def isPrim(x):
    for m in range(2, int(x / 2) + 1):
        if x % m == 0:
            return False
    return True

'''
def givePrimes(rang, start=0):
    if start > rang:
        raise ValueError("Invalid range")
    if start == rang:
        return []
    if start > 8:
        prim = []
    else:
        prim = [2, 3, 5, 7]
        if rang <= 8:
            return [i for i in prim if start <= i <= rang]
        else:
            prim = [2, 3, 5, 7]
            start = 8
    for n in range(start, rang + 1):
        if isPrim(n):
            prim.append(n)
    return prim
'''
def givePrimes(n):
    """ Input n>=6, Returns a list of primes, 2 <= p < n """
    n +=1
    n, correction = n - n % 6 + 6, 2 - (n % 6 > 1)
    sieve = [True] * (n // 3)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = [False] * ((n // 6 - k * k // 6 - 1) // k + 1)
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = [False] * ((n // 6 - k * (k - 2 * (i & 1) + 4) // 6 - 1) // k + 1)
    return [2, 3] + [3 * i + 1 | 1 for i in range(1, n // 3 - correction) if sieve[i]]

def int_to_bin(x, l=8):
    return f'{x:0{l}b}'


def isPalindrom(data):
    if type(data) == int:
        if is_even(len(str(data))):
            if split(data)[0] == rotate(split(data)[1]):
                return True
            return False
        else:
            data = str(data)
            m = math.ceil(len(data) / 2)
            deleteMid(data)
            return data
    if type(data) == str:
        return
    return


def split(data):
    data = str(data)
    x = len(data)
    return [data[:x // 2], data[x // 2:x]]


def rotate(data):  # rotating number and return list
    data = str(data)
    out = []
    for n in data:
        out.insert(0, n)
    return listToString(out)


def deleteMid(data):
    data = str(data)
    if len(data) % 2 == 1:
        data[int(math.ceil(len(data) / 2))]
    return data


def matrToText(data):
    out = ""
    for m in data:
        out += "(\t"
        for e in m:
            out += F"{e}\t"
        out += ")\n"
    return out


def create_random_matrix(h, l, ran=10):
    return [[random.randint(0, ran) for n in range(l)] for n in range(h)]


def create_empty_matrix(h, l):
    return [[0] * l for n in range(h)]


def total(numbers):
    return sum(numbers)


def mean(numbers):
    return sum(numbers) / len(numbers)


def median(numbers):
    numbers.sort()
    if len(numbers) % 2:
        mid = len(numbers) // 2
        median = numbers[mid]
    else:
        mid_right = len(numbers) // 2
        mid_left = mid_right - 1
        median = (numbers[mid_right] + numbers[mid_left]) / 2
    return median


def deleteLastLine():
    print("\033[A                             \033[A")


def onQuit():
    while True:
        z = getpass(prompt="Do you want to quit [y, n]: ")
        if z == "y":
            deleteLastLine()
            print("Good bye")
            return True
        if z == "n":
            deleteLastLine()
            return False
        deleteLastLine()


def distinct_prime_factors(x, prims=[]):
    s = 0
    last = 0
    out = []
    if not prims:
        s = 0
    else:
        s = prims[-1]
        if prims.count(0)>0:
            prims.remove(0)
    prim = prims + givePrimes(int(x), start=s)
    while x > 1:
        for p in prim:
            if x % p == 0:
                out.append(p)
                x /= p
                break
    return out


def list_remove_duplicates(x):
    return list(set(x))


def printMatr(matr):
    for yL in matr:
        print(yL)
    print("------")
    return


def get_diagonals_of_matrix(x, diag=[], rings=1):
    size = len(x)
    if not diag:
        for n in range(0, int((size - 1) / 2)):
            diag.append(x[n][n])
            diag.append(x[(-1 * n) - 1][n])
            diag.append(x[n][(-1 * n) - 1])
            diag.append(x[(-1 * n) - 1][(-1 * n) - 1])
        return diag
    else:
        for n in range(0, rings):
            diag.append(x[n][n])
            diag.append(x[(-1 * n) - 1][n])
            diag.append(x[n][(-1 * n) - 1])
            diag.append(x[(-1 * n) - 1][(-1 * n) - 1])
        return diag


def calc_diagonals_of_matrix(x):
    # calc diagonals
    size = len(x)
    amount = 1
    for n in range(0, int((size - 1) / 2)):
        amount += x[n][n]
        # print(F"+ {matr[n][n]}")
        amount += x[(-1 * n) - 1][n]
        # print(F"+ {matr[(-1*n)-1][n]}")
        amount += x[n][(-1 * n) - 1]
        # print(F"+ {matr[n][(-1*n)-1]}")
        amount += x[(-1 * n) - 1][(-1 * n) - 1]
        # print(F"+ {matr[(-1*n)-1][(-1*n)-1]}")
        # print("------")
    return amount


def create_spiral_matrix_rb(maxS=3):
    matr = [[1]]
    size = 1
    posX = 0
    posY = 0
    dir = ""
    print(maxS*maxS)
    for n in range(1, (maxS*maxS)+1):
        add = (size - 1) / 2
        if n > math.pow(size, 2):
            size += 2
            if size > maxS:
                size -= 2
                break
            for i in matr:
                i.insert(0, 0)
                i.append(0)
            matr.insert(0, emptyList(size))
            matr.append(emptyList(size))
            # printMatr(matr)
            add = (size - 1) / 2
            # print(add)
            posX += 1
            matr[int(posY + add)][int(posX + add)] = n
            dir = "south"
            # printMatr(matr)
            continue
        if posX > 0 and abs(posX) * 2 + 1 == size:  # rechts unten
            if posY > 0 and abs(posY) * 2 + 1 == size:
                dir = "west"
        if posX < 0 and abs(posX) * 2 + 1 == size:  # links unten
            if posY > 0 and abs(posY) * 2 + 1 == size:
                dir = "north"
        if posX < 0 and abs(posX) * 2 + 1 == size:  # links oben
            if posY < 0 and abs(posY) * 2 + 1 == size:
                dir = "east"
        if dir == "south":
            posY += 1
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        if dir == "west":
            posX -= 1
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        if dir == "north":
            posY -= 1
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        if dir == "east":
            posX += 1
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        # printMatr(matr)
    return matr


def create_spiral_matrix_rt(maxS=3):
    matr = [[1]]
    size = 1
    posX = 0
    posY = 0
    dir = ""
    #print(maxS * maxS)
    for n in range(1, (maxS * maxS) + 1):  #für jede zahl die vorkommt
        add = (size - 1) / 2
        if n > math.pow(size, 2):  #wenn größer als aktuelle matrix
            size += 2  #erhöhe größe
            #if size > maxS:
            #    size -= 2
            #    break
            for i in matr:  #vorne und hinten 0 hinzufügen in jeder list / erweitern
                i.insert(0, 0)
                i.append(0)
            matr.insert(0, emptyList(size)) #erweitern/neue listen adden
            matr.append(emptyList(size))
            # printMatr(matr)
            add = (size - 1) / 2
            # print(add)
            posX += 1
            matr[int(posY + add)][int(posX + add)] = n
            #print(F"y:{posY}+{add}][x:{posX}+{add}")
            dir = "north"
            #printMatr(matr)    #visualize creation
            continue
        #print("size", size)
        if posX > 0 and abs(posX) * 2 + 1 == size:  # rechts
            if posY < 0 and abs(posY) * 2 + 1 == size:  #oben
                dir = "west"
                #print("west")
        if posX < 0 and abs(posX) * 2 + 1 == size:  # links
            if posY > 0 and abs(posY) * 2 + 1 == size:  #unten
                dir = "east"
        if posX < 0 and abs(posX) * 2 + 1 == size:  # links
            if posY < 0 and abs(posY) * 2 + 1 == size:  #oben
                dir = "south"
        if dir == "south":
            posY += 1
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        if dir == "west":
            posX -= 1
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        if dir == "north":
            posY -= 1
            #print(F"y{posY}+{add}][x{posX}+{add}")
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        if dir == "east":
            posX += 1
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        #printMatr(matr)            #visualize creation
    return matr

def create_spiral_matrix_rt_add_ring(matr=[[1]], addi=1):
    size = len(matr)
    posX = (size - 1) / 2
    posY = (size - 1) / 2
    dir = ""
    #print(maxS * maxS)
    for n in range(matr[len(matr)-1][len(matr)-1]+1, ((len(matr)+2*addi)*(len(matr)+2*addi)) + 1):#für jede zahl die vorkommt
        #print(n, size)
        add = (size - 1) / 2
        if n > math.pow(size, 2):  #wenn größer als aktuelle matrix
            size += 2  #erhöhe größe
            #if size > maxS:
            #    size -= 2
            #    break
            for i in matr:  #vorne und hinten 0 hinzufügen in jeder list / erweitern
                i.insert(0, 0)
                i.append(0)
            matr.insert(0, emptyList(size)) #erweitern/neue listen adden
            matr.append(emptyList(size))
            # printMatr(matr)
            add = (size - 1) / 2
            # print(add)
            posX += 1
            matr[int(posY + add)][int(posX + add)] = n
            #print(F"[y:{posY}+{add}][x:{posX}+{add}]={n}")
            dir = "north"
            #printMatr(matr)    #visualize creation
            continue
        #print("size", size)
        if posX > 0 and abs(posX) * 2 + 1 == size:  # rechts
            if posY < 0 and abs(posY) * 2 + 1 == size:  #oben
                dir = "west"
                #print("west")
        if posX < 0 and abs(posX) * 2 + 1 == size:  # links
            if posY > 0 and abs(posY) * 2 + 1 == size:  #unten
                dir = "east"
        if posX < 0 and abs(posX) * 2 + 1 == size:  # links
            if posY < 0 and abs(posY) * 2 + 1 == size:  #oben
                dir = "south"
        if dir == "south":
            posY += 1
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        if dir == "west":
            posX -= 1
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        if dir == "north":
            posY -= 1
            #print(F"y{posY}+{add}][x{posX}+{add}")
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        if dir == "east":
            posX += 1
            matr[int(posY + add)][int(posX + add)] = n
            # printMatr(matr)
        #printMatr(matr)            #visualize creation
    return matr


def permustations(x):
    """
    hello world

    :param x: integer
    :return: int
    """
    return permutations(x)

def main():
    print(isprim2(12))

if __name__ == '__main__':
    time = datetime.now()
    main()
    print(datetime.now() - time)