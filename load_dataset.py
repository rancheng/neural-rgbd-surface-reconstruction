from load_scannet import load_scannet_data, load_tum_data


def load_dataset(args):

    if args.dataset_type == "scannet":
        images, depth_images, poses, hwf, frame_indices = load_scannet_data(basedir=args.datadir,
                                                                            trainskip=args.trainskip,
                                                                            downsample_factor=args.factor,
                                                                            translation=args.translation,
                                                                            sc_factor=args.sc_factor,
                                                                            crop=args.crop)

        print('Loaded scannet', images.shape, hwf, args.datadir)

    # Calls to other dataloaders go here
    # elif args.dataset_type == "":

    elif args.dataset_type == "tum":
        print('Loading tum_dataset...')
        images, depth_images, poses, hwf, frame_indices = load_tum_data(basedir=args.datadir,
                                                                            trainskip=args.trainskip,
                                                                            downsample_factor=args.factor,
                                                                            translation=args.translation,
                                                                            sc_factor=args.sc_factor,
                                                                            crop=args.crop)
        print('Loaded tum_dataset', args.dataset_type, 'exiting')
    else:
        print("not supported dataset")
        return
    return images, depth_images, poses, hwf, frame_indices
