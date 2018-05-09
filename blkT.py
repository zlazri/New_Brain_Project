import numpy as np

def blkT(arr1):

    '''Description: Transposes 2x2 blocks of a 2 channel array.

       Input:

       arr1: 2xN array, where N is an even number greater than 2.

             OR

             Nx2 array, where N is an even number greater than 2.

       arr1 = [ B(1)  B(2) ...  B(N/2)]

             OR

            = [ B(1) ]
              [ B(2) ]
              [   .  ]
              [   .  ]
              [   .  ]
              [B(N/2)]
       
              [0]
       arr2 = [0]
              [.]
              [.]
              [.]
              [0]

             OR

              [  0  0  ...  0  ]


       Output: arr2, whose block elements are now the tranpose of arr1

              [ B(1).T ]
       arr2 = [ B(2).T ]
              [   .    ]
              [   .    ]
              [   .    ]
              [B(N/2).T]

            OR

              [B(1).T  B(2).T  ...  B(N/2).T]
    '''

    sz = arr1.shape
    arr2 = np.zeros((sz[1], sz[0]))
    assert(sz[0]%2 == 0 and sz[1]%2 == 0), "Dimension Error: Input dimensions must be even"
    assert(sz[0] == 2 or sz[1] == 2), "Dimension Error: input dimensions must be 2xN or Nx2"

    if sz[0] == 2:
        for i in range(sz[1]):
            if i%2 == 0:
                arr2[i,0] = arr1[0,i]
                arr2[i,1] = arr1[0,i+1]
            else:
                arr2[i,0] = arr1[1,i-1]
                arr2[i,1] = arr1[1,i]
            
    elif sz[1] == 2:
        for i in range(sz[0]):
            if i%2 == 0:
                arr2[0,i] = arr1[i,0]
                arr2[0,i+1] = arr1[i,1]
            else:
                arr2[1,i-1] = arr1[i,0]
                arr2[1,i] = arr1[i,1]

    return arr2
