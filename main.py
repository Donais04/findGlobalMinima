from algs import *
import itertools
from concurrent.futures import ProcessPoolExecutor

_FILE_TO_READ = "lessNAD+.mol"




#minimize recombination [1,2,3] mutation [1,2,3]
#[[['recombination', 1], ['mutation', 1]],[['recombination', 1], ['mutation', 2]]]
noWrite = False
algs = []
debug = False
if __name__ == "__main__":
    alg = str(sys.argv[1])
    kiss = []
    wish = []
    repeat = 1
    batch = ""
    stage = 2
    type = ""
    randomIt = True
    for i in sys.argv[2:]:
      if i[0] == "-":
        kiss.append([])
        wish.append(i[1:])
      else:
        kiss[-1].append(i)
    i = 0
    print(kiss)
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
      elif wish[i] == 'nw':
        noWrite = True
        kiss.pop(i)
        wish.pop(i)
        i += -1
      elif wish[i] == 'stage':
        stage = int(kiss.pop(i)[0])
        wish.pop(i)
        i += -1
      elif wish[i] == 'type':
        if kiss[i][0].lower() == "xyz":
          type = "XYZ"
        elif kiss[i][0].lower() == "pym":
          type = "PYM"
        elif kiss[i][0].lower() == "ff":
          type = "FF"
        else:
          raise KeyError("type must be xyz, pym, or ff")
        kiss.pop(i)
        wish.pop(i)
        i += -1
      elif wish[i] == 'disp':
        kiss[i] = [len(kiss[i]) == 0 or kiss[i][0].lower()[0] == "t"]
      elif wish[i] == 'nr':
        kiss.pop(i)
        wish.pop(i)
        randomIt = False
      elif wish[i] == 'db':
        kiss.pop(i)
        wish.pop(i)
        debug = True
      i += 1
    kiss1= []
    for i in range(len(kiss)):
      if not(kiss[i] == []):
        kiss1.append(kiss[i])
    kiss = kiss1
    print("Simulating all combinations between " + str(kiss) + ("" if repeat == 1 else (" " + str(repeat) + " times")))
    miss = list(itertools.product(*kiss))
    print("miss ", miss)
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
    print(liss)
    count = 0
    for _ in range(repeat):
      for i in liss:
        count += 1
        myMol = molocule()
        myMol.molToMine(_FILE_TO_READ)
        if randomIt:
          myMol.rando()
        algs.append(Algorithm(myMol, alg, i, id = count, b=batch, stage=stage, listType=type, rand=randomIt))

def f(alg:Algorithm) -> int:
    #print("here")
    if not(debug):
      alg.run()
    else:
      while(True):
        do = input("What would you like to do to the molocule? 0: print as mol, 1: edit list, 2: print as type, 3: randomize, 4: exit")
        try:
          if do == "0":
            print(alg.mol.mineToMol())
          elif do == "1":
            l = alg.listOutFunc()
            print(l)
            do1 = input("input your index then the number to change to")
            l[int(do1.split(" ")[0])] = float(do1.split(" ")[1])
            alg.listInFunc(l)
          elif do == "2":
            print(alg.listOutFunc())
          elif do == "4":
            exit(0)
        except:
          print("error, try again")
    if not(noWrite):
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