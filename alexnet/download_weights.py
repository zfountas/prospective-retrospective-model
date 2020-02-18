address = 'http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy'

import urllib.request
urllib.request.urlretrieve(address, 'bvlc_alexnet.npy')
print("Done")
