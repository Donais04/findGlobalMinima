from algs import *
import itertools
import sys

_FILE_TO_READ = "lessNAD+.mol"

#TODO add nr to pym


#minimize recombination [1,2,3] mutation [1,2,3]
#[[['recombination', 1], ['mutation', 1]],[['recombination', 1], ['mutation', 2]]]
noWrite = False
algs = []
debug = False


def displayHelp(alg: str = ""):
    # Known algorithms (from algs.py)
    alg_info = {
      "minimize": {
          "desc": "Local gradient-based optimization using scipy.optimize.minimize.",
          "params": {
              "method": {
                  "desc": "The minimization method used",
                  "options": "Nelder-Mead, Powell, CG, BFGS, Newton-CG, L-BFGS-B, TNC, COBYLA, COBYQA, SLSQP, trust-constr, dogleg, trust-ncg, trust-exact, trust-krylov",
                  "type": "string",
                  "default": "L-BFGS-B"
              },
              "tol": {
                  "desc": "Tolerance for termination",
                  "type": "float",
                  "default": "1e-2"
              },
              "maxiter": {
                  "desc": "Maximum number of iterations",
                  "type": "int",
                  "default": "1000"
              }
          }
      },

      "basinhopping": {
          "desc": "Global optimization via basin-hopping (random perturbations + local minimization).",
          "params": {
              "T": {
                  "desc": "Temperature parameter controlling acceptance of uphill moves",
                  "type": "float",
                  "default": "100.0"
              },
              "niter_success": {
                  "desc": "Stop if no improvement after this many iterations",
                  "type": "int",
                  "default": "5"
              },
              "disp": {
                  "desc": "Display status messages",
                  "type": "bool",
                  "default": "False"
              }
          }
      },

      "differential_evolution": {
          "desc": "Population-based global optimization algorithm.",
          "params": {
              "maxiter": {
                  "desc": "Maximum number of generations",
                  "type": "int",
                  "default": "1000"
              },
              "popsize": {
                  "desc": "Population size multiplier",
                  "type": "int",
                  "default": "10"
              },
              "mutation": {
                "desc": "Mutation constant",
                "type": "float",
                "default": "0.5"
            },
            "recombination": {
                "desc": "Recombination constant",
                "type": "float",
                "default": "0.7"
            },
            "tol": {
                "desc": "Relative tolerance for convergence",
                "type": "float",
                "default": "1e-3"
            },
            "strategy": {
                "desc": "Evolution strategy",
                "options": "best1bin, best1exp, rand1bin, rand1exp, rand2bin, rand2exp",
                "type": "string",
                "default": "best1bin"
            }
        }
    },

      "dual_annealing": {
        "desc": "Simulated annealing-based global optimization.",
        "params": {
            "maxiter": {
                "desc": "Maximum number of iterations",
                "type": "int",
                "default": "1000"
            },
            "initial_temp": {
                "desc": "Initial temperature",
                "type": "float",
                "default": "5230"
            },
            "restart_temp_ratio": {
                "desc": "Temperature ratio for restart",
                "type": "float",
                "default": "2e-5"
            },
            "visit": {
                "desc": "Parameter controlling visiting distribution",
                "type": "float",
                "default": "2.62"
            },
            "accept": {
                "desc": "Parameter controlling acceptance distribution",
                "type": "float",
                "default": "-5.0"
            },
            "maxfun": {
                "desc": "Maximum number of function evaluations",
                "type": "int",
                "default": "1e7"
            },
            "no_local_search": {
                "desc": "Disable local search phase",
                "type": "bool",
                "default": "False"
            }
        }
    },

      "random_restart": {
        "desc": "Repeated local minimization from random starting points.",
        "params": {
            "iter": {
                "desc": "Number of random restarts",
                "type": "int",
                "default": "20"
            },
            "mini_iter": {
                "desc": "Iterations for each local minimization",
                "type": "int",
                "default": "100"
            },
            "disp": {
                "desc": "Display progress messages",
                "type": "bool",
                "default": "False"
            }
        }
    },

      "brute_force": {
        "desc": "Random sampling within bounds (no gradient use).",
        "params": {
            "iter": {
                "desc": "Number of random samples",
                "type": "int",
                "default": "1000"
            }
        }
    },

      "direct": {
        "desc": "DIRECT global optimization algorithm.",
        "params": {
            "eps": {
                "desc": "Desired relative error in solution",
                "type": "float",
                "default": "0.0001"
            },
            "maxiter": {
                "desc": "Maximum number of iterations",
                "type": "int",
                "default": "1000"
            },
            "locally_biased": {
                "desc": "Use locally biased variant",
                "type": "bool",
                "default": "True"
            },
            "vol_tol": {
                "desc": "Volume tolerance for stopping",
                "type": "float",
                "default": "1e-16"
            },
            "len_tol": {
                "desc": "Length tolerance for stopping",
                "type": "float",
                "default": "1e-6"
            }
        }
    },

      "shgo": {
        "desc": "Simplicial homology global optimization.",
        "params": {
            "n": {
                "desc": "Number of sampling points",
                "type": "int",
                "default": "100"
            },
            "iters": {
                "desc": "Number of refinement iterations",
                "type": "int",
                "default": "1"
            },
            "sampling_method": {
                "desc": "Sampling method used",
                "options": "simplicial, sobol",
                "type": "string",
                "default": "simplicial"
            }
        }
      }
      }

    global_flags = {
        "-f FILE": "Input molecule file (default: lessNAD+.mol)",
        "-r N": "Repeat simulation N times",
        "-batch NAME": "Batch label for output grouping",
        "-nw": "Disable writing output files",
        "-stage N": "Set stage (1 or 2)",
        "-type [xyz|pym|ff]": "Coordinate representation",
        "-nr": "Disable random initialization",
        "-db": "Enable debug mode",
        "-skip": "Skip stage 1 preprocessing"
    }

    def print_param(name, meta):
        line = f"  -{name} ({meta.get('type', 'unknown')})"
        
        if "default" in meta:
            line += f", default={meta['default']}"
        
        print(line)
        print(f"      {meta.get('desc', '')}")
        
        if "options" in meta:
            print(f"      options: {meta['options']}")
        print()

    if alg == "":
        print("\n=== GENERAL HELP ===\n")
        print("Usage:")
        print("  python main.py ALGORITHM [options]")
        print("  python main.py -help")
        print("  python main.py ALGORITHM -help\n")

        print("Available Algorithms:")
        for a, info in alg_info.items():
            print(f"  - {a}: {info['desc']}")

        print("\nGlobal Options:")
        for k, v in global_flags.items():
            print(f"  {k:20} {v}")

        print("\nExample:")
        print("  python main.py minimize -f molecule.mol -r 5 -type xyz\n")

    else:
        if alg not in alg_info:
            print(f"Unknown algorithm '{alg}'. Use -help to list valid options.")
            return

        info = alg_info[alg]

        print(f"\n=== HELP: {alg.upper()} ===\n")
        print(info["desc"])

        print("\nAlgorithm Parameters:\n")
        if info["params"]:
            for pname, meta in info["params"].items():
                print_param(pname, meta)
        else:
            print("  (No parameters)")

        print("\n\nUsage Example:")
        example = f"python main.py {alg}"
        for pname in list(info["params"].keys())[:2]:  # show a couple params
            example += f" -{pname} VALUE"
        print(f"  {example}")

        print("\nGlobal Options (also supported):")
        for k, v in global_flags.items():
            print(f"  {k:20} {v}")

        print()
    exit()

if __name__ == "__main__":
    alg = str(sys.argv[1])
    if alg == "-help":
      displayHelp()
    elif str(sys.argv[2]) == "-help":
      displayHelp(alg)
    kiss = []
    wish = []
    repeat = 1
    batch = ""
    stage = 2
    type = "XYZ"
    randomIt = True
    skip = False
    for i in sys.argv[2:]:
      if i[0] == "-":
        kiss.append([])
        wish.append(i[1:])
      else:
        kiss[-1].append(i)
    i = 0
    #print("wish", wish, "kiss", kiss)
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
        i += -1
      elif wish[i] == 'db':
        kiss.pop(i)
        wish.pop(i)
        debug = True
        i += -1
      elif wish[i] == 'skip':
        kiss.pop(i)
        wish.pop(i)
        skip = True
        i += -1
      i += 1
    kiss1= []
    for i in range(len(kiss)):
      if not(kiss[i] == []):
        kiss1.append(kiss[i])
    kiss = kiss1
    print("Simulating all combinations between " + str(kiss) + ("" if repeat == 1 else (" " + str(repeat) + " times")))
    miss = list(itertools.product(*kiss))
    #print("miss ", miss)
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
        if randomIt:
          myMol.rando()
        algs.append(Algorithm(myMol, alg, i, id = count, b=batch, stage=stage, listType=type, rand=randomIt, skip1=skip))

def f(alg:Algorithm) -> int:
    #print("here")
    if not(debug):
      alg.run()
    else:
      while(True):
        do = input("What would you like to do to the molocule? 0: print as mol, 1: edit list, 2: print as type, 3: randomize, 4: perform stage 1, 5: run, 6: edit settings, 7: exit")
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
          elif do == "3":
            alg.mol.rando()
          elif do == "4":
            print("starting at", alg.mol.scoreFull())
            newAlg = Algorithm(alg.mol,stage=1,listType="XYZ")
            newAlg.run()
            alg.mol.listToMineXYZ(newAlg.mol.mineToListXYZ())
            print("done to score", alg.mol.scoreFull())
          elif do == '5':
            alg.run()
          elif do == '6':
            setting = input("what setting do you want to change from the following: " + str(list(alg.kwargs.keys())))
            change = input("and what do you want to change it to? It is currently at " + str(alg.kwargs[setting]))
            try:
              if "." in change:
                alg.kwargs[setting] = float(change)
              else:
                alg.kwargs[setting] = int(change)
            except:
              alg.kwargs[setting] = change
          elif do == "7":
            exit(0)
        except Exception as e:
          print("error:",e)
    if not(noWrite):
      alg.save()
    print(i)
    return 0
  
for i in algs:
  f(i)
