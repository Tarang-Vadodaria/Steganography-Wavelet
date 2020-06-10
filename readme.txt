There are 2 images X and Y 
X - To send 
Y - To hide 

Process to merge X in Y to get Z (look alike of  X)

1. RGB Split(X), RGB Split(Y)
2. DWT(X), DWT(Y)
3. SVD(X), SVD(Y)
4. Embed(X,Y)

Process to extract X from Z

1. RGB Split(Z)
2. DWT(Z)
3. SVD(Z)
4. Extract(X)