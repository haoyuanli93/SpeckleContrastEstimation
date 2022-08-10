def plot_contrast(data, run, epix, q, beam, weights=None):
    '''
    data is in {'kbar', 'beta'}
    beam is in {'vcc','cc','both'}
    #* (beta < 1) * (beta > -1) * (p1 != 0) * (p2 != 0)
    '''

    prob_name = 'run{:04d}_epx{}_q{}.npz'.format(run, epx, q)
    path = '/cds/home/t/tanduc/Masks_Tan-Duc/may2k22/contrast_files/'
    npz = np.load(path + prob_name)
    p1 = npz['p1']
    kbar = npz['kbar']
    beta = npz['beta']

    file_path = '/cds/data/drpsrcf/XPP/xppc00120/scratch/hdf5/smalldata/'
    fn = 'run{:04d}_epx{}.h5'.format(run, epix)

    with h5py.File(file_path + fn, 'r') as f:
        vcc = np.array(f['vcc'])
        cc = np.array(f['cc'])
        i3 = np.array(f['i3'])

    mean_i3 = np.mean(i3)
    std_i3 = np.std(i3)
    n_beta = len(beta)
    n_vcc = len(vcc)

    vcc = vcc[:n_beta]
    cc = cc[:n_beta]
    i3 = i3[:n_beta]
    both = ((vcc > 4) * (cc > 4)).astype('int')
    x = np.arange(1, len(cc) + 1, 1)

    if beam == 'vcc':

        events = (vcc > 4) * (both == 0) * (i3 >= mean_i3 - std_i3) * (i3 <= mean_i3 + std_i3) * (p1 != 0)
        if data == 'kbar':

            new_data = np.array([kbar[i] for i in range(len(events)) if events[i]])
            avg, err = cum_avg(new_data)
            print('kbar: %.7f' % avg[-1])
        elif data == 'beta':
            new_data = np.array([beta[i] for i in range(len(events)) if events[i]])
            avg, err = cum_avg(new_data)
            print('contrast: %.4f' % avg[-1])
        else:
            print('choose a correct data type')

    elif beam == 'cc':
        events = (cc > 4) * (both == 0) * (i3 >= mean_i3 - std_i3) * (i3 <= mean_i3 + std_i3) * (p1 != 0)

        if data == 'kbar':
            new_data = np.array([kbar[i] for i in range(len(events)) if events[i]])
            avg, err = cum_avg(new_data)
            print('kbar: %.7f' % avg[-1])
        elif data == 'beta':
            new_data = np.array([beta[i] for i in range(len(events)) if events[i]])
            avg, err = cum_avg(new_data)
            print('contrast: %.4f' % avg[-1])
        else:
            print('choose a correct data type')

    elif beam == 'both':
        events = (both == 1)  # * (i3 >= mean_i3 - std_i3) * (i3 <= mean_i3 + std_i3) * (p1 != 0)

        if data == 'kbar':
            new_data = np.array([kbar[i] for i in range(len(kbar)) if events[i]])
            avg, err = cum_avg(new_data)
            print('kbar: %.7f' % avg[-1])
        elif data == 'beta':
            new_data = np.array([beta[i] for i in range(len(beta)) if events[i]])
            avg, err = cum_avg(new_data)
            print('contrast: %.4f' % avg[-1])
        else:
            print('choose a correct data type')

    else:
        print('choose a correct beam configuration')

    print(len(events[events]))
    fig = plt.figure(figsize=(9, 9))
    plt.plot(new_data, '.', color='navy', alpha=0.5)
    plt.plot(avg, color='black')
    plt.fill_between(np.arange(len(avg)), avg - err, avg + err, color='crimson', alpha=0.3)
    plt.xlabel('event #')
    plt.ylabel(data)
    if data == 'beta':
        plt.ylim(-1, 3)
        plt.title('Contrast for run{:04d} and epix{}'.format(run, epix))
    elif data == 'kbar':
        plt.title('Kbar for run{:04d} and epix{}'.format(run, epix))
    plt.show()

    plt.clf()
    return None
