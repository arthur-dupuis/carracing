from treat_screen import transform_obs

def next_action(observation):
    road_data = transform_obs(observation)
    l = 0
    r = 0
    for i in range(len(road_data[66])):
        if (i<48):
            l += road_data[66][i]
        else:
            r += road_data[66][i]
    if l>r :
        return 'left'
    else :
        return 'right'