'''Augment CKY and Cell classes with pretty-printing functionality
'''

# Two CKY methods
def CKY_pprint(self,cell_width=8):
    '''Try to print matrix in a nicely lined-up way'''
    row_max_height=[0]*(self.n)
    col_max_width=[0]*(self.n)
    print_matrix=[]
    for r in range(self.n-1):
         # rows
         row=[]
         for c in range(r+1,self.n):
             # columns
             if c>r:
                 # This is one we care about, get a cell form
                 #  and tabulate width, height and update maxima
                 cf=self.matrix[r][c].str(cell_width)
                 nlines=len(cf)
                 if nlines>row_max_height[r]:
                     row_max_height[r]=nlines
                 if cf!=[]:
                     nchars=max(len(l) for l in cf)
                     if nchars>col_max_width[c]:
                         col_max_width[c]=nchars
                 row.append(cf)
         print_matrix.append(row)
    row_fmt='|'.join("%%%ss"%col_max_width[c] for c in range(1,self.n))
    row_index_len=len(str(self.n-2))
    row_index_fmt="%%%ss"%row_index_len
    row_div=(' '*(row_index_len+1))+(
        '+'.join(('-'*col_max_width[c]) for c in range(1,self.n)))
    print( (' '*(row_index_len+1))+(' '.join(str(c).center(col_max_width[c])
                   for c in range(1,self.n))))
    for r in range(self.n-1):
        if r!=0:
            print( row_div)
        mrh=row_max_height[r]
        for l in range(mrh):
            print( row_index_fmt%(str(r)) if l==int(mrh/2) else ' ',end=' ')
            row_strs=['' for c in range(r)]
            row_strs+=[wtp(l,print_matrix[r][c],mrh) for c in range(self.n-(r+1))]
            print( row_fmt%tuple(row_strs))

def CKY_log(self,message,*args,**kwargs):
    if self.verbose:
        print( ' '*kwargs.get('indent',0)+(message%args))

# A utility function
def wtp(l,subrows,maxrows):
    '''figure out what row or filler from within a cell
    to print so that the printed cell fills from
    the bottom.  l will be in range(mrh)'''
    offset=maxrows-len(subrows)
    if l>=offset:
        return subrows[l-offset]
    else:
        return ''

# Three Cell methods

def Cell__str__(self):
    return self.str()

def Cell_str(self,width=8):
    '''Try to format labels in a rectangle,
    aiming for max-width as given, but only
    breaking between labels'''
    syms=self.labels()
    res=[]
    i=0
    line=[]
    ll=-1
    for i,s in enumerate(syms):
        s=str(s)
        m=len(s)
        if ll+m>width and ll!=-1:
            res.append(' '.join(line))
            line=[]
            ll=-1
        line.append(s)
        ll+=m+1
        i=i+1
    if ll==-1:
        return res
    res.insert(0,' '.join(line))
    return res

def Cell_log(self,message,*args,**kwargs):
    self.matrix.log("%s,%s: "+message,self._row,self._column,*args,**kwargs)

