'''Completed version of cky.py for ANLP 2016 assignment 2: CKY parser'''
import sys,re
import nltk
from collections import defaultdict
import itertools
import cfg_fix
from cfg_fix import parse_grammar, CFG, Tree
from nltk.grammar import Nonterminal
from pprint import pprint
# The printing and tracing functionality is in a separate file in order
#  to make this file easier to read
from cky_print import CKY_pprint, CKY_log, Cell__str__, Cell_str, Cell_log

class CKY:
    '''An implementation of a Cocke-Kasami-Younger (bottom-up) CFG parser.

    Goes beyond strict CKY\'s insistance on Chomsky Normal Form.
    It allows arbitrary unary productions, not just NT->T
    ones, that is X -> Y with either Y -> A B or Y -> Z .
    It also allows mixed binary productions, that is NT -> NT T or -> T NT'''

    def __init__(self,grammar):
        '''Create an extended CKY parser for a particular grammar

        Grammar is an NLTK CFG
        consisting of unary and binary rules (no empty rules,
        no more than two symbols on the right-hand side
        (We use "symbol" throughout this code to refer to _either_ a string or
        an nltk.grammar.Nonterminal, that is, the two thinegs we find in
        nltk.grammar.Production)

        :type grammar: nltk.grammar.CFG, as fixed by cfg_fix
        :param grammar: A context-free grammar'''

        self.verbose=False
        assert isinstance(grammar,CFG)
        self.grammar=grammar
        # split and index the grammar
        self.buildIndices(grammar.productions())

    def buildIndices(self,productions):
        '''Build indices to the productions passed in.
        
        Splits into unary (allowing symbol as RHS)
        and binary (allowing pairs of symbols as RHS).
        In both cases the _value_ is augmented with the LHS,
        so always a Nonterminal.

        :type productions: list(nltk.grammar.Production)
        :param productions: CFG rules
        :return: None, but creates values for two instance variables,
                 unary (type dict(str|nltk.grammar.Nonterminal=>
                                  list(nltk.grammar.Nonterminal)))
                 and binary, (type dict(tuple(str|nltk.grammar.Nonterminal)=>
                                        list(nltk.grammar.Nonterminal)))
        '''
        self.unary=defaultdict(list) # Indexed by RHS, a symbol
        self.binary=defaultdict(list) # Indexed by RHS, a pair of symbols
        for production in productions:
            # get the two parts of the production
            rhs=production.rhs()
            lhs=production.lhs()
            assert (len(rhs)>0 and
                    len(rhs)<=2) # enforce the length restriction
            if len(rhs)==1:
                self.unary[rhs[0]].append(lhs) # index on the RHS symbol itself
            else:
                self.binary[rhs].append(lhs) # index on the RHS pair, which
                                             #  is a 2-tuple

    def parse(self,tokens,verbose=False):
        '''Initialise a matrix from the token list, then try to parse it.

        Uses the pre-indexed grammar 

        :type tokens: list(str)
        :param tokens: the tokens of the candidate sentence
        :type verbose: bool
        :param verbose: show debugging output if True, defaults to False
        :rtype: bool|int
        :return: True if our grammar start symbol occurs in the top
                  right-hand corner after processing finishes,
                  False otherwise
        '''
        self.verbose=verbose
        self.words = tokens
        self.n = len(self.words)+1 # because we number the 'spaces'
                                   #  _between_ the words
        self.matrix = []
        # We index by row, then column
        #  So Y below is 1,2 and Z is 0,3
        #    1   2   3  ...
        # 0  X   X   Z
        # 1      Y   X
        # 2          X
        # ...
        for r in range(self.n-1):
             # rows, indexed by the point _before_ a word
             row=[]
             for c in range(self.n):
                 # columns, indexed by the point _after_ a word
                 if c>r:
                     # This is one we care about, on or above main diagonal
                     #  so we add a Cell instance here
                     row.append(Cell(r,c,self))  # tell the Cell where it is
                                                 #  and its containing matrix
                 else:
                     # just a filler, never looked at
                     row.append(None)
             self.matrix.append(row)
        self.unaryFill() # fill in the main diagonal
        self.binaryScan() # then scan the rest of the top-right half,
                           #  in length order of the corresponding substring
        # We win if the start symbol of the gramamr
        #  is in the top-right corner
        return self.matrix[0][self.n-1].hasCat(self.grammar.start())

    def unaryFill(self):
        '''Fill in the main diagonal of a CKY matrix

        Start with words from the input, and look them up in the
        unary productions of our grammar
        '''
        for r in range(self.n-1):
            # for each cell on the main diagonal
            cell=self.matrix[r][r+1]
            # interpret it as the word _between_ input positions r and r+1
            word=self.words[r]
            label=TerminalLabel(word) # make a simple Label
            # add the word
            cell.addLabel(label)

    def binaryScan(self):
        '''Fill in the upper-right diagonals of the matrix

        Proceed left to right within length order .
        Effectively dynamic programming in increasing order of
          constituent length
        '''
        for span in range(2, self.n):
            # We start with constituents of length 2,
            #  because unaryFill did length 1
            for start in range(self.n-span):
                # there are n-2 possible places for a length 2 constituent,
                #  n-3 for one of length 3, etc.
                end = start + span
                for mid in range(start+1, end):
                    # over all the possible intermediate points of
                    #  a span from start to end,
                    #  see if we can build something
                    self.maybeBuild(start, mid, end)

    def maybeBuild(self, start, mid, end):
        '''
        Build any constituents we can from start->end

        First combine every constituent that runs from start->mid
         with every one that runs from mid->end, and check that pair
         in the index of binary productions in our grammar

        :type start: int
        :param start: index of left end of possible new constituent
        :type mid: int
        :param mid: index of boundary between candidate pair of substrings
        :type end: int
        :param end: index of right end of possible new constituent
        '''
        self.log("%s--%s--%s:",start, mid, end)
        cell=self.matrix[start][end] # this is where our results will go
        left=self.matrix[start][mid] # this has possible first parts of a pair
        right=self.matrix[mid][end] # and this has the second parts
        for lcat in left.symbols(): # symbols we have in left cell
            for rcat in right.symbols(): # symbols we have in right cell
                # lcat, rcat are all possible pairs of symbols
                if (lcat,rcat) in self.binary:
                    # OK, we have some new constituents
                    for nt in self.binary[(lcat,rcat)]:
                        self.log("%s -> %s %s", nt, lcat, rcat, indent=1)
                        # nt will be each possible symbol for a new spanning
                        #  constituent
                        label=BinaryNTLabel(nt,lcat,left,rcat,right)
                        cell.addLabel(label)

    def firstTree(self,symbol=None):
        '''Put together a Tree rooted here for our 'first' parse

        :type symbol: nltk.grammar.Nonterminal
        :param symbol: Find a parse rooted in this, defaults to
                        the start symbol of our grammar
        :rtype: nltk.Tree, as modified by cfg_fix
        :return: The first parse for symbol
        '''
        if symbol is None:
            symbol=self.grammar.start()
        # Top right-hand corner covers the whole string
        return self.matrix[0][self.n-1].trees(symbol,True)

    def allTrees(self,symbol=None):
        '''Put together Trees for all parses rooted here

        :type symbol: nltk.grammar.Nonterminal
        :param symbol: Find parses rooted in this, defaults to
                        the start symbol of our grammar
        :rtype: list(nltk.Tree), as modified by cfg_fix
        :return: parses for symbol
        '''
        if symbol is None:
            symbol=self.grammar.start()
        # See if the top-right cell has a tree with the appropriate label
        return self.matrix[0][self.n-1].trees(symbol,False)

# helper methods from cky_print
CKY.pprint=CKY_pprint
CKY.log=CKY_log

class Cell:
    '''A cell in a CKY matrix'''
    def __init__(self,row,column,matrix):
        '''A new Cell

        :type row: int
        :param row: row index of this Cell in matrix
        :type column: int
        :param column: column index of this Cell in matrix
        :type matrix: list(list(Cell|None)))
        :param matrix: The CKY matrix we\'re a part of
        '''
        
        # We keep track of row and col only for display purposes
        self._row=row
        self._column=column
        self.matrix=matrix # This gives access to grammar and verbosity
        # We index all the labels by their symbol (aka category)
        # In other words, _labels.keys() will be a set (i.e. no
        #  duplicates) equivalent to what the _labels themselves are
        #  in the original simpler CKY recogniser.
        self._labels={}

    def __repr__(self):
        '''A simple string representation of the cell'''
        return "<Cell %s,%s>"%(self._row,self._column)

    def addLabel(self,label):
        '''Add a Label to this Cell
        :type label: Label
        :param label: the label to be added
        '''
        assert isinstance(label,Label)
        symbol=label.symbol()
        if symbol in self._labels:
            # It's old, just add as another possible parse
            self.log("found another %s",symbol)
            self._labels[symbol].append(label)
        else:
            # New, so start a list of possible parses keyed by symbol
            self._labels[symbol]=[label]
            # and propagate upward from its label by checking unary rules
            self.unaryUpdate(label,1)

    def symbols(self):
        '''symbols of all constituents here
        :rtype: list(str|Nonterminal)
        :return: one per Label'''
        return self._labels.keys()

    def labels(self):
        '''labels of all constituents here

        :rtype: list(Label)
        :return: the cell contents'''
        # Only used in output at the moment...
        return itertools.chain(*(iter(labs) for labs in self._labels.values()))

    def labelsFor(self,symbol):
        '''The labels we have for a specified symbol.
        Will throw a KeyError if asked about an unknown symbol

        :type symbol: Nonterminal|str
        :param symbol: symbol to look for in the labels of this cell
        :rtype: list(Label)
        :return: all labels for symbol
        '''
        # Return the labels for a given symbol
        return self._labels[symbol]

    def hasCat(self,symbol):
        '''Check for a symbol among known labels here

        :type symbol: Nonterminal|str
        :param symbol: symbol to look for in the labels of this cell
        :rtype: bool
        :return: True iff a constituent of the given category is found here
        '''
        return symbol in self._labels

    def unaryUpdate(self,label,depth=0,recursive=False):
        '''Apply any matching unary rules recursively
        
        We\'ve just added label here: if its
        symbol is found in unaries (a dictionary of LHS symbols
        indexed by RHS symbol) add new parent Labels here and try
        ourselves recursively for them

        :type label: Label
        :param label: Label just added to this Cell
        :type depth: int
        :param depth: logging indentation control, reflects recursion depth
        :type recursive: bool
        :param recursive: True iff called from itself, for logging
        '''
        symbol=label.symbol()
        if not recursive:
            # first time, log position in matrix and symbol
            self.log("%s",str(symbol),indent=depth)
        if symbol in self.matrix.unary:
            # symbol is the RHS of at least one production
            for parent in self.matrix.unary[symbol]:
                # for each LHS
                # log the production
                self.matrix.log("%s -> %s",
                                parent,symbol,indent=depth+1)
                # add the lhs as a new unary label
                self.addLabel(UnaryNTLabel(parent,symbol,self))
                # the above will recurse iff parent is new

    def trees(self,symbol,justOne):
        '''Build one or all Trees rooted here for symbol

        Cardinality is controlled by justOne

        :type symbol: nltk.grammar.Nonterminal
        :param symbol: Find parse(s) rooted in this
        :type justOne: bool
        :param justOne: one Tree if True, otherwise all Trees
        :rtype: nltk.Tree|list(nltk.Tree), as modified by cfg_fix
        :return: parse(s) for symbol
        '''
        roots=self.labelsFor(symbol)
        if justOne:
            # Just build a tree using the first label in the list of labels
            #  for this category.
            #  It will be the first one found, which in turn
            #  will be the one with the simplest (fewest unary branches)
            #  and/or shortest (left child is shorter than all others)
            #  This results in a maximally right-branching tree
            return roots[0].trees(True)
        else:
            # Concatenate all the lists of all possible subtrees
            #  for all the labels we have here
            # Note that 'sum' concatenates trees
            return sum((root.trees(False) for root in roots),[])

# helper methods from cky_print
Cell.__str__=Cell__str__
Cell.str=Cell_str
Cell.log=Cell_log

# We try to be as efficient as possible in representing the results
#  of a parse.
# So we don't build complete trees until we're asked for them,
#  (at which point we build them all -- a generator would be even
#  better).
# The key point is that non-terminals only _point_ to their child/children,
#  so in particular if even if both children of a binary label
#  are ambiguous, the storage cost per node is constant at parse time.
# Only as the trees are built does the cost of cross-products begin to
#  be felt.
# This is a version of an approach known as a 'packed forest', that is, a
#  compact representation of a _set_ of trees.

class Label:
    '''A label for a substring in a CKY chart Cell

    Includes a terminal or non-terminal symbol, possibly other
    information.'''
    def __init__(self,symbol):
        '''Create a Label from a symbol

        :type symbol: str|nltk.grammar.Nonterminal
        :param symbol: a terminal or non-terminal
        '''
        assert (isinstance(symbol,Nonterminal) or
                isinstance(symbol,str))
        self._symbol=symbol

    def __str__(self):
        '''A prettier short version of ourselves -- just our cat'''
        return str(self.symbol())

    def symbol(self):
        return self._symbol

class TerminalLabel(Label):
    # No __init__, as Label.__init__ does all we need
    def __eq__(self,other):
        '''test for equality

        other must be a Label,
        and symbols have to be equal

        :type other: Label
        :param other: candidate for equality
        :rtype: bool
        :return: True if symbols are equal
        '''
        # Note that this is stricter than you will sometimes see:
        #  we could just return False for other not a Label
        assert isinstance(other,Label)
        # OTOH we _do_ anticipate legitimate comparison with other
        #  kinds of Labels (although note that as written this parser
        #  _never_ actually compares any kind of Labels for equality)
        return (isinstance(other,TerminalLabel) and
                self._symbol==other._symbol)

    def __repr__(self):
        '''A simple string representation of a terminal Label'''
        return "<Label %s>"%self._symbol

    def trees(self,justOne):
        '''a leaf

        :type justOne: bool
        :param justOne: one Tree if True, otherwise all Trees
        :rtype: str|[str]
        :return: A single leaf, possibly as a singleton list
        '''
        if justOne:
            # We just use our symbol
            return self.symbol()
        else:
            # Need a list, because higher up it will be iterated over
            return [self.symbol()]

class UnaryNTLabel(Label):
    def __init__(self,cat,lCat=None,lCell=None):
        '''Create a UnaryNTLabel from its symbol and child

        :type cat: nltk.grammar.Nonterminal
        :param symbol: our non-terminal category
        :type lCat: nltk.grammar.Nonterminal
        :param lCat: the category of our (packed) child
        :type lCell: Cell
        :param lCell: the Cell where our (packed) child is found
        '''
        assert isinstance(cat,Nonterminal)
        assert (isinstance(lCat,Nonterminal) or
                isinstance(lCat,str))
        assert isinstance(lCell,Cell)
        # Parent initialisation gets us started
        Label.__init__(self,cat)
        # Save our own instance vars
        self._lCat=lCat
        self._lCell=lCell

    def __eq__(self,other):
        '''test for equality

        :type other: Label
        :param other: candidate for equality
        :rtype: bool
        :return: True iff self and other count as equal

        For equality other must be a UnaryNTLabel and
        respective symbol and child cell and symbol have to be equal
        '''
        # Note that this is stricter than you will sometimes see:
        #  we could just return False for other not a Label
        assert isinstance(other,Label)
        # OTOH we _do_ anticipate legitimate comparison with other
        #  kinds of Labels (although note that as written this parser
        #  _never_ actually compares any kind of Labels for equality)
        return (isinstance(other,UnaryNTLabel) and
                self._symbol==other._symbol and
                self._lCat==other._lCat and
                self._lCell==other._lCell)

    def __repr__(self):
        '''A simple string representation of a unary NT Label'''
        return "<Label %s%s>"%(self._symbol,
                             (" %s@%s"%(self._lCat,repr(self._lCell))))

    def trees(self,justOne):
        '''New Tree(s) with ourselves as the parent to
           the first/all of the Trees from our lCell labelled with lCat

        :type justOne: bool
        :param justOne: one Tree if True, otherwise all Trees
        :rtype: nltk.Tree|list(nltk.Tree), as modified by cfg_fix
        :return: Tree(s) rooted here, down to the leaves
        '''
        if justOne:
            # We just need one Tree from our (left) sub-cell
            #  to make the single child of a new Tree with our category
            res=Tree(self.symbol(),[self._lCell.trees(self._lCat,True)])
        else:
            # We need to build one Tree with our category
            #  for every possible subTree
            #  from our (left and only) sub-cell
            res=[Tree(self.symbol(),[tree])
                 for tree in self._lCell.trees(self._lCat,False)]
        return res

class BinaryNTLabel(Label):
    def __init__(self,cat,lCat=None,lCell=None,rCat=None,rCell=None):
        '''Create a BinaryNTLabel from a symbol and its children

        :type cat: nltk.grammar.Nonterminal
        :param symbol: our non-terminal category
        :type lCat: nltk.grammar.Nonterminal
        :param lCat: the category of our (packed) left child
        :type lCell: Cell
        :param lCell: the Cell where our (packed) left child is found
        :type rCat: nltk.grammar.Nonterminal
        :param rCat: the category of our (packed) left child
        :type rCell: Cell
        :param rCell: the Cell where our (packed) left child is found
        '''
        assert isinstance(cat,Nonterminal)
        assert (isinstance(lCat,Nonterminal) or
                isinstance(lCat,str))
        assert isinstance(lCell,Cell)
        assert (isinstance(rCat,Nonterminal) or
                isinstance(rCat,str))
        assert isinstance(rCell,Cell)
        # Parent initialisation gets us started
        Label.__init__(self,cat)
        # Save our own instance vars
        self._lCat=lCat
        self._lCell=lCell
        self._rCat=rCat
        self._rCell=rCell

    def __repr__(self):
        '''A simple string representation of a binary Label'''
        return "<Label %s%s%s>"%(self._symbol,
                             (" %s@%s"%(self._lCat,repr(self._lCell))),
                             (" %s@%s"%(self._rCat,repr(self._rCell))))

    def __eq__(self,other):
        '''test for equality

        :type other: Label
        :param other: candidate for equality
        :rtype: bool
        :return: True iff self and other count as equal

        For equality other must be a BinaryNTLabel and
        respective symbol and both left and right child cell and
        symbol have to be equal
        '''
        # Note that this is stricter than you will sometimes see:
        #  we could just return False for other not a Label
        assert isinstance(other,Label)
        # OTOH we _do_ anticipate legitimate comparison with other
        #  kinds of Labels (although note that as written this parser
        #  _never_ actually compares any kind of Labels for equality)
        return (isinstance(other,BinaryNTLabel) and
                self._symbol==other._symbol and
                self._lCat==other._lCat and
                self._rCat==other._rCat and
                self._lCell==other._lCell and
                self._rCell==other._rCell)

    def trees(self,justOne):
        '''New Tree(s) with ourselves as the parent to
           the first/all of the Trees
           from our [lr]Cell labelled with our [lr]Cat

        :type justOne: bool
        :param justOne: one Tree if True, otherwise all Trees
        :rtype: nltk.Tree|list(nltk.Tree), as modified by cfg_fix
        :return: Tree(s) rooted here, down to the leaves
        '''
        if justOne:
            # We need one Tree each from our left and right sub-cells,
            #  allowing us to build a new Tree with our category
            #  as node label and those Trees as children
            res=Tree(self.symbol(),[self._lCell.trees(self._lCat,True),
                                   self._rCell.trees(self._rCat,True)])
        else:
            # We need to build one Tree with our category
            #  for every possible pair of subTrees
            #  from our left and right sub-cells
            # This is where we pay the cross-product price
            res=[Tree(self.symbol(),[lTree,rTree])
                 for lTree in self._lCell.trees(self._lCat,False)
                 for rTree in self._rCell.trees(self._rCat,False)]
        return res

def tokenise(tokenstring):
    '''Split a string into a list of tokens

    We treat punctuation as
    separate tokens, and split contractions into their parts.

    So for example "I'm leaving." --> ["I","'m","leaving","."]
      
    @type tokenstring: str
    @param tokenstring the string to be tokenised
    @rtype: list(str)
    @return: the tokens found in tokenstring'''

    # Note that we do _not_ split on word-internal hyphens, and do
    #  _not_ attempt to diagnose or repair end-of-line hyphens
    # Nor do we attempt to distinguish the use of full-stop to mark
    #  abbreviations from its end-of-sentence use, or the use of single-quote
    #  for possessives from its use for contractions and quotations (for which
    #  the following arguably does the wrong thing.
    return re.findall(
        # We use three sub-patterns:
        #   one for words and the first half of possessives
        #   one for the rest of possessives
        #   one for punctuation
        r"[-\w]+|'\w+|[^-\w\s]+",
        tokenstring,
        re.U # Use unicode classes, otherwise we would split
             # "são jaques" into ["s", "ão","jaques"]
        )
