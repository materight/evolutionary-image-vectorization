
class RouletteWheelSelection(): 
    pass

class RankBasedSelection(): 
    pass

class TruncatedSelection(): 
    def __init__(self, selection_cutoff):
        self.selection_cutoff = selection_cutoff

class TournamentSelection(): 
    def __init__(self, k):
        self.k = k 