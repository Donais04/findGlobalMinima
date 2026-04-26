from algs import *
import itertools
from concurrent.futures import ProcessPoolExecutor

_FILE_TO_READ = "lessNAD+.mol"




#minimize recombination [1,2,3] mutation [1,2,3]
#[[['recombination', 1], ['mutation', 1]],[['recombination', 1], ['mutation', 2]]]
algs = []
if __name__ == "__main__":
    alg = str(sys.argv[1])
    kiss = []
    wish = []
    repeat = 1
    batch = ""
    for i in sys.argv[2:]:
      if i[0] == "-":
        kiss.append([])
        wish.append(i[1:])
      else:
        kiss[-1].append(i)
    i = 0
    while i < len(wish):
      if wish[i] == 'r':
        repeat = int(kiss.pop(i)[0])
        wish.pop(i)
        i += -1
      elif wish[i] == 'batch':
        batch = kiss.pop(i)[0]
        wish.pop(i)
        i += -1
      elif wish[i] == 'f':
        _FILE_TO_READ = kiss.pop(i)[0]
        wish.pop(i)
        i += -1
      i += 1
    print("Simulating all combinations between " + str(kiss) + ("" if repeat == 1 else (" " + str(repeat) + " times")))
    miss = list(itertools.product(*kiss))
    liss = []
    for i in range(len(miss)):
      builder = []
      for j in range(len(miss[i])):
        try:
          if "." in miss[i][j]:
            builder.append([wish[j],float(miss[i][j])])
          else:
            builder.append([wish[j],int(miss[i][j])])
        except:
          builder.append([wish[j],miss[i][j]])
      liss.append(builder)
    #print(liss)
    count = 0
    for _ in range(repeat):
      for i in liss:
        count += 1
        myMol = molocule()
        myMol.molToMine(_FILE_TO_READ)
        myMol.rando()
        algs.append(Algorithm(myMol, alg, i, id = count))

def f(alg:Algorithm) -> int:
    #print("here")
    alg.run()
    alg.save()
    print(i)
    return 0
  
for i in algs:
  f(i)
#if __name__ == '__main__':
#
#    with ProcessPoolExecutor(max_workers=4) as executor:
#        var = executor.map(f, algs)
#    input("Press enter key when all is finished")