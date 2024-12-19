import pandas as pd
import numpy as np

""" all info from here https://www.iro.umontreal.ca/~labimage/Dataset/technicalReport.pdf """

def load_delays():
    """Load camera delays for each scenario"""
    # Delays from Figure 3 table

    delays = {
        1: [3, 3, 8, 4, 23, 6, 6, 0],
        2: [25, 40, 0, 16, 18, 33, 33, 6],
        3: [12, 16, 8, 16, 35, 20, 20, 0],
        4: [72, 79, 78, 0, 68, 82, 83, 56],
        5: [17, 24, 5, 11, 18, 7, 26, 28, 0],
        6: [0, 100, 106, 90, 89, 103, 104, 89],
        7: [28, 14, 16, 0, 1, 17, 18, 20],
        8: [92, 79, 0, 81, 64, 81, 82, 56],
        9: [18, 9, 1, 19, 13, 11, 12, 0],
        10: [14, 15, 19, 33, 12, 17, 19, 0],
        11: [23, 4, 20, 14, 0, 6, 7, 12],
        12: [21, 6, 13, 8, 0, 3, 7, 0],
        13: [16, 33, 0, 7, 27, 27, 36, 13],
        14: [49, 36, 38, 0, 29, 29, 7, 14],
        15: [15, 19, 19, 15, 34, 40, 23, 0],
        16: [23, 29, 0, 2, 12, 9, 3, 3],
        17: [21, 26, 15, 0, 10, 0, 29, 18],
        18: [99, 105, 86, 0, 84, 108, 109, 77],
        19: [19, 27, 16, 19, 5, 29, 0, 20],
        20: [25, 9, 3, 10, 10, 4, 5, 0],
        21: [20, 30, 22, 3, 8, 33, 32, 0],
        22: [0, 46, 51, 41, 53, 46, 47, 34],
        23: [31, 52, 52, 45, 54, 60, 50, 0],
        24: [3, 36, 7, 0, 37, 10, 33, 1]
    }

    return delays

def load_ground_truth():
    """Load ground truth data with position codes"""
    # Example format: [scenario, camera_ref, start_frame, end_frame, position_code]
    ground_truth = [
        # Scenario 1
        [1, 11, 874, 1011, 1],
        [1, 11, 1012, 1079, 6],
        [1, 11, 1080, 1108, 2],
        [1, 11, 1109, 1285, 3],
        
        # Scenario 2
        [2, 4, 308, 374, 1],
        [2, 4, 375, 399, 2],
        [2, 4, 400, 600, 3],
        
        # Scenario 3
        [3, 11, 380, 590, 1],
        [3, 11, 591, 625, 2],
        [3, 11, 626, 784, 3],
        
        # Scenario 4
        [4, 6, 230, 287, 1],
        [4, 6, 288, 314, 2],
        [4, 6, 315, 380, 3],
        [4, 6, 381, 600, 6],
        [4, 6, 601, 638, 2],
        [4, 6, 639, 780, 3],
        
        # Scenario 5
        [5, 11, 288, 310, 1],
        [5, 11, 311, 336, 2],
        [5, 11, 337, 450, 3],
        
        # Scenario 6
        [6, 1, 325, 582, 1],
        [6, 1, 583, 629, 2],
        [6, 1, 630, 750, 3],
        
        # Scenario 7
        [7, 6, 330, 475, 1],
        [7, 6, 476, 507, 2],
        [7, 6, 508, 680, 3],
        
        # Scenario 8
        [8, 4, 144, 270, 1],
        [8, 4, 271, 298, 2],
        [8, 4, 299, 380, 3],
        
        # Scenario 9
        [9, 11, 310, 472, 1],
        [9, 11, 473, 505, 5],
        [9, 11, 506, 576, 7],
        [9, 11, 577, 627, 6],
        [9, 11, 628, 651, 2],
        [9, 11, 652, 760, 3],
        
        # Scenario 10
        [10, 11, 315, 461, 1],
        [10, 11, 462, 511, 5],
        [10, 11, 512, 530, 2],
        [10, 11, 531, 680, 3],
        
        # Scenario 11
        [11, 7, 378, 463, 1],
        [11, 7, 464, 489, 2],
        [11, 7, 490, 600, 3],
        
        # Scenario 12
        [12, 11, 355, 604, 1],
        [12, 11, 605, 653, 2],
        [12, 11, 654, 770, 3],
        
        # Scenario 13
        [13, 4, 301, 430, 1],
        [13, 4, 431, 476, 5],
        [13, 4, 477, 525, 7],
        [13, 4, 526, 636, 5],
        [13, 4, 637, 717, 8],
        [13, 4, 718, 780, 6],
        [13, 4, 781, 822, 6],
        [13, 4, 823, 863, 2],
        [13, 4, 864, 960, 3],
        
        # Scenario 14
        [14, 6, 372, 555, 1],
        [14, 6, 556, 590, 5],
        [14, 6, 591, 856, 8],
        [14, 6, 857, 934, 6],
        [14, 6, 935, 988, 6],
        [14, 6, 989, 1023, 2],
        [14, 6, 1024, 1115, 3],
        
        # Scenario 15
        [15, 11, 363, 486, 1],
        [15, 11, 487, 530, 5],
        [15, 11, 531, 630, 7],
        [15, 11, 631, 754, 6],
        [15, 11, 755, 787, 2],
        [15, 11, 788, 870, 3],
        
        # Scenario 16
        [16, 4, 380, 455, 1],
        [16, 4, 456, 488, 5],
        [16, 4, 489, 530, 4],
        [16, 4, 531, 568, 6],
        [16, 4, 569, 629, 5],
        [16, 4, 630, 645, 4],
        [16, 4, 646, 670, 6],
        [16, 4, 671, 731, 5],
        [16, 4, 732, 817, 7],
        [16, 4, 818, 890, 6],
        [16, 4, 891, 940, 2],
        [16, 4, 941, 1000, 3],
        
        # Scenario 17
        [17, 6, 251, 315, 1],
        [17, 6, 316, 340, 5],
        [17, 6, 341, 361, 4],
        [17, 6, 362, 388, 6],
        [17, 6, 389, 410, 5],
        [17, 6, 411, 430, 4],
        [17, 6, 431, 460, 6],
        [17, 6, 461, 531, 5],
        [17, 6, 532, 620, 7],
        [17, 6, 621, 729, 6],
        [17, 6, 730, 770, 2],
        [17, 6, 771, 860, 3],
        
        # Scenario 18
        [18, 6, 301, 378, 1],
        [18, 6, 379, 430, 5],
        [18, 6, 431, 530, 7],
        [18, 6, 531, 570, 6],
        [18, 6, 571, 601, 2],
        [18, 6, 602, 740, 3],
        
        # Scenario 19
        [19, 10, 255, 498, 1],
        [19, 10, 499, 600, 2],
        [19, 10, 601, 770, 3],
        
        # Scenario 20
        [20, 11, 301, 544, 1],
        [20, 11, 545, 672, 2],
        [20, 11, 673, 800, 3],
        
        # Scenario 21
        [21, 11, 408, 537, 1],
        [21, 11, 538, 608, 5],
        [21, 11, 609, 794, 7],
        [21, 11, 795, 863, 6],
        [21, 11, 864, 901, 2],
        [21, 11, 902, 1040, 3],
        
        # Scenario 22
        [22, 1, 317, 586, 1],
        [22, 1, 587, 685, 5],
        [22, 1, 686, 737, 7],
        [22, 1, 738, 766, 6],
        [22, 1, 767, 808, 2],
        [22, 1, 809, 930, 3],
        
        # Scenario 23 (long sequence)
        [23, 11, 393, 662, 1],
        [23, 11, 663, 688, 5],
        [23, 11, 689, 710, 4],
        [23, 11, 711, 744, 6],
        [23, 11, 745, 1519, 1],
        [23, 11, 1520, 1595, 2],
        [23, 11, 1596, 1661, 6],
        [23, 11, 1662, 1730, 1],
        [23, 11, 1731, 1769, 5],
        [23, 11, 1770, 1839, 4],
        [23, 11, 1840, 1886, 6],
        [23, 11, 1887, 2645, 1],
        [23, 11, 2646, 2698, 5],
        [23, 11, 2699, 2958, 8],
        [23, 11, 2959, 3035, 6],
        [23, 11, 3036, 3156, 1],
        [23, 11, 3157, 3237, 5],
        [23, 11, 3238, 3416, 8],
        [23, 11, 3417, 3573, 6],
        [23, 11, 3574, 3614, 2],
        [23, 11, 3615, 3745, 6],
        [23, 11, 3746, 3795, 5],
        [23, 11, 3796, 4042, 4],
        [23, 11, 4043, 4105, 6],
        [23, 11, 4106, 4204, 1],
        [23, 11, 4205, 4264, 5],
        [23, 11, 4265, 4440, 7],
        [23, 11, 4441, 4527, 6],
        [23, 11, 4528, 5200, 1],
        
        # Scenario 24
        [24, 6, 350, 974, 1],
        [24, 6, 975, 1315, 1],
        [24, 6, 1316, 1351, 5],
        [24, 6, 1352, 1414, 4],
        [24, 6, 1415, 1450, 6],
        [24, 6, 1451, 1750, 1],
        [24, 6, 1751, 1805, 5],
        [24, 6, 1806, 1844, 4],
        [24, 6, 1845, 1884, 6],
        [24, 6, 1885, 2490, 1],
        [24, 6, 2491, 2514, 5],
        [24, 6, 2515, 2563, 4],
        [24, 6, 2564, 2587, 6],
        [24, 6, 2588, 3040, 1],
        [24, 6, 3041, 3077, 5],
        [24, 6, 3078, 3125, 6],
        [24, 6, 3126, 3243, 1],
        [24, 6, 3244, 3353, 1],
        [24, 6, 3354, 3401, 5],
        [24, 6, 3402, 3500, 4]
    ]

    return pd.DataFrame(ground_truth, columns=['scenario', 'camera_ref', 'start_frame', 'end_frame', 'position_code'])

def position_to_binary(position_code):
    """Convert position codes to binary fall detection labels
    Returns 1 for fall (position_code == 2), 0 for non-fall"""
    return 1 if position_code == 2 else 0

def adjust_frames_for_delay(frame, scenario, camera, delays):
    """Adjust frame numbers based on camera delays"""
    if scenario in delays and camera <= len(delays[scenario]):
        return frame - delays[scenario][camera-1]
    return frame

def generate_fall_labels():
    """Generate fall detection labels with camera delays"""
    delays = load_delays()
    ground_truth = load_ground_truth()
    
    fall_labels = []
    
    # Process each scenario
    for scenario in ground_truth['scenario'].unique():
        scenario_data = ground_truth[ground_truth['scenario'] == scenario]
        
        # Process each camera (1-8)
        for camera in range(1, 9):
            # Get all periods for this scenario
            periods = scenario_data.sort_values('start_frame')
            
            # Process each period
            for _, period in periods.iterrows():
                # Adjust frames for camera delay
                start_frame = adjust_frames_for_delay(period['start_frame'], scenario, camera, delays)
                end_frame = adjust_frames_for_delay(period['end_frame'], scenario, camera, delays)
                
                # Convert position to binary label
                label = position_to_binary(period['position_code'])
                
                # Add to results if frames are valid
                if start_frame >= 0 and end_frame >= 0:
                    fall_labels.append([
                        scenario,          # chute (scenario number)
                        camera,            # cam
                        int(start_frame),  # start
                        int(end_frame),    # end
                        label             # label (0=non-fall, 1=fall)
                    ])
    
    # Convert to DataFrame and sort
    labels_df = pd.DataFrame(fall_labels, columns=['chute', 'cam', 'start', 'end', 'label'])
    labels_df = labels_df.sort_values(['chute', 'cam', 'start']).reset_index(drop=True)
    
    # Combine consecutive rows with the same label
    combined_labels = []
    prev_row = None
    
    for _, row in labels_df.iterrows():
        if prev_row is not None and row['label'] == prev_row['label'] and row['start'] == prev_row['end'] + 1:
            # Combine with previous row
            prev_row['end'] = row['end']
        else:
            if prev_row is not None:
                combined_labels.append(prev_row)
            prev_row = row
    
    if prev_row is not None:
        combined_labels.append(prev_row)
    
    combined_labels_df = pd.DataFrame(combined_labels, columns=['chute', 'cam', 'start', 'end', 'label'])
    
    return combined_labels_df


def save_labels(labels_df, output_file='fall_detection_labels_combines.csv'):
    """Save labels to CSV file"""
    labels_df.to_csv(output_file, index=False)
    print(f"Labels saved to {output_file}")

if __name__ == "__main__":
    # Generate and save labels
    labels_df = generate_fall_labels()
    save_labels(labels_df)
    
    # Display first few labels
    print("\nFirst few labels:")
    print(labels_df.head(10))

