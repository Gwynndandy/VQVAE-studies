import src.autoencoders as ac
import src.cnns as cnn
import src.training as trn
import src.data_handling as data
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--vqvae", help="Path to config of VQ-VAE")
    parser.add_argument("-v", "--vae", help="Path to config of VAE")
    parser.add_argument("-r", "--roi", help="Path to config of ROI")
    parser.add_argument("-t", "--test", help="System-check")
    parser.add_argument("-n", "--runs", help="Number of random runs")
    args = parser.parse_args()

    if args.vqvae:
        vq_config = ac.read_yaml_vqvae(args.vqvae)
    if args.vae:
        vae_config = ac.read_yaml_autoencoder(args.vae)
    if args.roi:
        roi_config = cnn.read_yaml_roi(args.roi)

    if args.vqvae:
        model,opt,sched,history = ac.get_vqvae(vq_config)
        train,test,val = data.make_np_ds(vq_config,"AE")
        path = r"/home/maniacalm/bin/VQVAE_studies/run_data/test"
        csv_filename,path = trn.init(history,"test",path)
        trn.vq_train(model,sched,opt,history,path,vq_config["num_epochs"],train,val,csv_filename)
        trn.vq_eval(model,opt,path,test,history)

    if args.vae:
        model,opt,sched,history = ac.get_vae(vae_config)
        train,test,val = data.make_np_ds(vae_config,"AE")
        path = r"/home/maniacalm/bin/VQVAE_studies/run_data/test"
        csv_filename,path = trn.init(history,"test",path)
        trn.vae_train(model,sched,opt,history,path,vae_config["num_epochs"],train,val,csv_filename)
        trn.vae_eval(model,opt,path,test,history)

    if args.roi:
        model,opt,sched,history = cnn.get_roi(roi_config)
        train,test,val = data.make_np_ds(roi_config,"ROI")
        path = r"/home/maniacalm/bin/VQVAE_studies/run_data/test"
        csv_filename,path = trn.init(history,"test",path)
        trn.cnn_train(model,sched,opt,history,path,roi_config["num_epochs"],train,val,csv_filename)
        trn.cnn_eval(model,opt,path,test,history)