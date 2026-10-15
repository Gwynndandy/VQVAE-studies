from ./src/autoencoders import get_vqvae,read_yaml_autoencoder
from ./src/training import init,train,eval
from ./src/data_handling import make_sc_ds

read_yaml_autoencoder($1)
get_vqvae(args)
make_sc_ds(args)
init(args)
train(args)
eval(args)