import numpy as np

labels_detection = np.genfromtxt("detection_labels.csv", dtype=int, delimiter=",")
ground_truth = np.genfromtxt("ground_truth_labels.csv", dtype=int, delimiter=",")

total_frames = len(ground_truth)

# # Calculate the confusion matrix
# confusion_mtx = np.zeros((4, 3, 3))
# for i in range(total_frames):
#     seats_gt = ground_truth[i, :]
#     seats_detection = labels_detection[i, :]
#     for seat in range(4):
#         confusion_mtx[seat, seats_gt[seat], seats_detection[seat]] += 1
# print(confusion_mtx)

# Calculate accuracy
correct = incorrect = 0
correct_soft = incorrect_soft = 0
SEEK_FRAMES = 51
for seat in range(4):
    this_seat_ground_truth = ground_truth[:, seat]
    this_seat_detection = labels_detection[:, seat]
    correct_mask = this_seat_ground_truth == this_seat_detection

    # Calculate hard accuracy
    this_seat_correct = np.sum(correct_mask)
    this_seat_incorrect = np.sum(~correct_mask)
    print("Hard accuracy for seat{}: {:.2f}".format(seat, this_seat_correct/total_frames*100))

    # Calculate soft accuracy
    this_seat_correct_soft = this_seat_incorrect_soft = 0
    for i in range(total_frames):
        gt, det = this_seat_ground_truth[i], this_seat_detection[i]
        if det == gt:
            this_seat_correct_soft += 1
        else:
            # Check if the state is detected within the buffer frames
            if np.any(this_seat_detection[i:i+SEEK_FRAMES] == gt):
                this_seat_correct_soft += 1
            else:
                this_seat_incorrect_soft += 1

    print("Soft accuracy for seat{}: {:.2f}".format(seat, this_seat_correct_soft/total_frames*100))     
    print("="*10)

    # Contribute to overall accuracy
    correct += this_seat_correct
    incorrect += this_seat_incorrect
    correct_soft += this_seat_correct_soft
    incorrect_soft += this_seat_incorrect_soft

print("Overall hard accuracy: {:.2f}".format(correct/(correct+incorrect)*100))
print("Overall soft accuracy: {:.2f}".format(correct_soft/(correct_soft+incorrect_soft)*100))