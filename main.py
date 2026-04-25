from algs import *


_FILE_TO_READ = "leastNAD+.mol"



myMol = molocule()
myMol.molToMine(open(_FILE_TO_READ).read())

algs = []
if __name__ == "__main__":
    t = int(sys.argv[1])
    print(t)
    for i in range(t):
      myMol = molocule()
      myMol.molToMine(open(_FILE_TO_READ).read())
      myMol.rando()
      algs.append(Algorithm(myMol,"differential_evolution"))
      
for i in algs:
  i.run()
  i.save()
  print(i)