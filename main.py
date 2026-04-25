from algs import *
import itertools

_FILE_TO_READ = "leastNAD+.mol"




#minimize recombination [1,2,3] mutation [1,2,3]
#[[['recombination', 1], ['mutation', 1]],[['recombination', 1], ['mutation', 2]]]
algs = []
if __name__ == "__main__":
    alg = str(sys.argv[1])
    kiss = []
    wish = []
    repeat = False
    for i in sys.argv[2:]:
      if i[0] == "-":
        if i[1:] == 'r':
          repeat = True
        else:
          kiss.append([])
          wish.append(i[1:])
      else:
        if repeat:
          builder = []
          for j in kiss[-1]:
            builder.append(j)
          kiss[-1] += builder
          repeat = False
        else:
          kiss[-1].append(i)
    print("Simulating all combinations between " + str(kiss))
    miss = list(itertools.product(*kiss))
    liss = []
    for i in range(len(miss)):
      builder = []
      for j in range(len(miss[i])):
        try:
          builder.append([wish[j],float(miss[i][j])])
        except:
          builder.append([wish[j],miss[i][j]])
      liss.append(builder)
    #print(liss)
    for i in liss:
      myMol = molocule()
      myMol.molToMine(_FILE_TO_READ)
      myMol.rando()
      algs.append(Algorithm(myMol, alg, i))

for i in algs:
  i.run()
  i.save()
  print(i)