runSilverBehenate = [242, ]

nameList = ["run", "Sample", "s4", "att", "CC", "VCC", "Cycling", "sd_delay"]

###################################################
#  The first batch of the data for H2O at 382 C
###################################################


runGroup1 = [
    [108, 'water', 0.15, 0.5, 17],
    [109, 'water', 0.15, 0.5, 17],
    [110, 'static', 0.15, 2.00E-02, 17],
    [111, 'static', 0.15, 2.00E-02, 17],
    [112, 'static', 0.15, 2.00E-02, 17],
    [113, 'water', 0.15, 0.2, 17],
    [114, 'water', 0.15, 0.4, 17],
    [115, 'water', 0.15, 0.4, 17],
    [118, 'static', 0.15, 2.00E-02, 8],
    [119, 'static', 0.15, 2.00E-02, 8],
    [120, 'static', 0.15, 2.00E-02, 8],
    [121, 'water', 0.15, 0.2, 8],
    [123, 'water', 0.15, 0.2, 8],
    [124, 'water', 0.15, 0.4, 8],
    [125, 'water', 0.15, 0.2, 7.1],
    [126, 'water', 0.15, 0.4, 7.1],
    [127, 'water', 0.15, 0.4, 7.1],
    [128, 'water', 0.15, 0.4, 7.1],
    [129, 'static', 0.15, 2.00E-02, 7.1],
    [130, 'static', 0.15, 2.00E-02, 7.1],
    [131, 'static', 0.15, 1.00E-02, 7.1],
    [132, 'static', 0.15, 2.00E-02, 9],
    [133, 'static', 0.15, 2.00E-02, 9],
    [134, 'static', 0.15, 1.00E-02, 9],
    [135, 'water', 0.15, 1.00E-05, 9],
    [136, 'water', 0.15, 0.2, 9],
    [137, 'water', 0.15, 0.4, 9],
    [138, 'water', 0.15, 0.2, 9],
    [139, 'static', 0.15, 2.00E-02, 9],
    [140, 'static', 0.15, 2.00E-02, 9],
    [141, 'static', 0.15, 1.00E-02, 9],
    [142, 'static', 0.15, 2.00E-02, 10],
    [143, 'static', 0.15, 2.00E-02, 10],
    [144, 'static', 0.15, 1.00E-02, 10],
    [145, 'water', 0.15, 1.00E-05, 10],
    [146, 'water', 0.15, 0.4, 10],
    [147, 'water', 0.15, 0.2, 10],
    [148, 'water', 0.15, 0.4, 10],
    [150, 'static', 0.15, 2.00E-02, 10],
    [153, 'static', 0.15, 1.00E-02, 10],
    [154, 'static', 0.15, 2.00E-02, 10],
    [156, 'static', 0.15, 1.00E-05, 11],
    [157, 'static', 0.15, 2.00E-02, 11],
    [158, 'static', 0.15, 2.00E-02, 11],
    [159, 'water', 0.15, 0.2, 11],
    [160, 'water', 0.15, 0.4, 11],
    [161, 'water', 0.15, 0.4, 11],
    [162, 'water', 0.15, 0.2, 11],
    [163, 'static', 0.15, 1.00E-05, 11],
    [164, 'static', 0.15, 2.00E-02, 11],
    [165, 'static', 0.15, 1.00E-02, 11],
    [166, 'static', 0.15, 1.00E-05, 12],
    [167, 'static', 0.15, 2.00E-02, 12],
    [168, 'static', 0.15, 2.00E-02, 12],
    [169, 'water', 0.15, 1.00E-05, 12],
    [170, 'water', 0.15, 0.2, 12],
    [171, 'water', 0.15, 0.4, 12],
    [172, 'water', 0.15, 0.4, 12],
    [173, 'static', 0.15, 0.4, 12],
    [174, 'static', 0.15, 2.00E-02, 12],
    [175, 'static', 0.15, 1.00E-02, 12],
    [176, 'static', 0.15, 1.00E-05, 13],
    [177, 'static', 0.15, 2.00E-02, 13],
    [178, 'static', 0.15, 2.00E-02, 13],
    [179, 'water', 0.15, 0.2, 13],
    [180, 'water', 0.15, 0.4, 13],
    [181, 'water', 0.15, 0.4, 13],
    [182, 'water', 0.15, 0.2, 13],
    [183, 'static', 0.15, 1.00E-05, 13],
    [184, 'static', 0.15, 2.00E-02, 13],
    [185, 'static', 0.15, 2.00E-02, 13],
    [186, 'static', 0.15, 1.00E-02, 14],
    [187, 'static', 0.15, 2.00E-02, 14],
    [188, 'static', 0.15, 2.00E-02, 14],
    [189, 'water', 0.15, 0.2, 14],
    [190, 'water', 0.15, 0.2, 14],
    [191, 'water', 0.15, 0.4, 14],
    [192, 'water', 0.15, 0.4, 14],
    [193, 'static', 0.15, 1.00E-02, 14],
    [194, 'static', 0.15, 2.00E-02, 14],
    [195, 'static', 0.15, 1.00E-02, 14],
    [196, 'static', 0.15, 2.00E-02, 15],
    [197, 'static', 0.15, 2.00E-02, 15],
    [198, 'static', 0.15, 1.00E-02, 15],
    [199, 'water', 0.15, 1.00E-02, 15],
    [200, 'water', 0.15, 0.4, 15],
    [201, 'water', 0.15, 0.4, 15],
    [202, 'water', 0.15, 0.2, 15],
    [203, 'static', 0.15, 1.00E-02, 15],
    [204, 'static', 0.15, 2.00E-02, 15],
    [205, 'static', 0.15, 1.00E-02, 15],
    [206, 'static', 0.15, 1.00E-02, 16],
    [207, 'static', 0.15, 2.00E-02, 16],
    [208, 'static', 0.15, 1.00E-02, 16],
    [209, 'water', 0.15, 0.2, 16],
    [210, 'water', 0.15, 0.4, 16],
    [211, 'water', 0.15, 0.4, 16],
    [212, 'water', 0.15, 0.2, 16],
    [213, 'static', 0.15, 5.00E-03, 16],
    [214, 'static', 0.15, 1.80E-02, 16],
    [215, 'static', 0.15, 1.00E-02, 16],
    [218, 'static', 0.15, 1.00E-02, 17],
    [219, 'static', 0.15, 2.00E-02, 17],
    [221, 'static', 0.15, 1.00E-02, 17],
    [222, 'water', 0.15, 0.2, 17],
    [223, 'water', 0.15, 0.4, 17],
    [224, 'water', 0.15, 0.4, 17],
    [225, 'water', 0.15, 2.00E-02, 17],
    [226, 'static', 0.15, 2.00E-02, 17],
    [227, 'static', 0.15, 2.00E-02, 17],
    [228, 'static', 0.15, 1.00E-02, 17],
    [229, 'static', 0.15, 3.00E-02, 8],
    [230, 'static', 0.15, 3.00E-02, 8],
    [231, 'static', 0.15, 1.50E-02, 8],
    [232, 'water', 0.15, 0.3, 8],
    [233, 'water', 0.15, 0.6, 8],
    [234, 'water', 0.15, 0.6, 8],
    [235, 'water', 0.15, 0.3, 8],
    [236, 'static', 0.15, 1.50E-02, 8],
    [237, 'static', 0.15, 2.50E-02, 8],
    [238, 'static', 0.15, 1.50E-02, 8],
    [248, 'static', 0.15, 3.00E-02, 8],
    [249, 'static', 0.15, 3.00E-02, 8],
    [250, 'static', 0.15, 1.00E-02, 8],
    [253, 'water', 0.15, 0.4, 8],
    [254, 'water', 0.15, 0.4, 8],
    [255, 'water', 0.15, 0.4, 8],
    [256, 'water', 0.15, 0.4, 8],
    [257, 'water', 0.15, 0.2, 8],
    [258, 'water', 0.15, 0.2, 8],
    [259, 'water', 0.15, 0.3, 8],
    [260, 'water', 0.15, 0.3, 8],
    [261, 'static', 0.15, 1.00E-02, 8],
    [262, 'static', 0.15, 1.00E-02, 8],
    [263, 'static', 0.15, 1.00E-02, 8],
    [264, 'static', 0.15, 1.00E-02, 8],
    [265, 'static', 0.15, 2.00E-02, 8],
    [266, 'static', 0.15, 2.00E-02, 8],
    [267, 'static', 0.15, 2.00E-02, 8],
    [268, 'static', 0.15, 4.30E-02, 8],
    [269, 'static', 0.15, 4.30E-02, 8],
    [270, 'static', 0.15, 2.00E-02, 8],
    [271, 'static', 0.15, 3.60E-02, 8],
    [272, 'static', 0.15, 3.60E-02, 8],
    [273, 'static', 0.15, 2.00E-02, 8],

]

runH2O_382_unfocused = [74, 76, 95]
runH2O_382_focused = {}
runSilica_forH2O_382 = {}
for entry in runGroup1:

    # Check if this is for water:
    if entry[1] == 'water':
        if "{:.1f}".format(entry[-1]) in runH2O_382_focused.keys():
            runH2O_382_focused["{:.1f}".format(entry[-1])].append(entry[0])
        else:
            runH2O_382_focused.update({"{:.1f}".format(entry[-1]): [entry[0], ]})

    elif entry[1] == 'static':
        if "{:.1f}".format(entry[-1]) in runSilica_forH2O_382.keys():
            runSilica_forH2O_382["{:.1f}".format(entry[-1])].append(entry[0])
        else:
            runSilica_forH2O_382.update({"{:.1f}".format(entry[-1]): [entry[0], ]})
