'''
This file contains all the possible different operators that can be sed to configure the behaviour of the Genetic Algorithm or Particle Swarm Optimization
'''

# Common parent class to handle class string conversion for printing
class Operator:
    def __init__(self, *params): self.params = [str(p) for p in params]
    def __str__(self): return type(self).__name__ + ('('+', '.join(self.params)+')' if len(self.params) > 0 else '')
        

# Parent selections
class selection:
    class RouletteWheelSelection(Operator): pass
    class RankBasedSelection(Operator): pass
    class TruncatedSelection(Operator): 
        def __init__(self, selection_cutoff):
            super(selection.TruncatedSelection, self).__init__(selection_cutoff)
            self.selection_cutoff = selection_cutoff
    class TournamentSelection(Operator): 
        def __init__(self, k):
            super(selection.TournamentSelection, self).__init__(k)
            self.selection_cutoff = k


# Survivor replacement
class replacement:
    class CommaReplacement(Operator): pass
    class PlusReplacement(Operator): pass
    class CrowdingReplacement(Operator):
        def __init__(self, pool_size):
            super(replacement.CrowdingReplacement, self).__init__(pool_size)
            self.pool_size = pool_size


# Crossover types
class crossover:
    class OnePointCrossover(Operator): pass
    class UniformCrossover(Operator): pass
    class ArithmeticCrossover(Operator): pass


# Velocity update rules
class velocity_update:
    class Standard(Operator): pass
    class FullyInformed(Operator): pass
    class ComprehensiveLearning(Operator): pass


# Neighbourhood topologies
class topology:
    class DistanceTopology(Operator): pass
    class RingTopology(Operator): pass
    class StarTopology(Operator): pass