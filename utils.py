from matplotlib import pyplot


def read_labels(label_file):
    labels = {}
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.split()
            file_name = fields[0]
            bottom, left = int(fields[1]), int(fields[2])
            top, right = int(fields[3]), int(fields[4])
            if file_name not in labels:
                labels[file_name] = []
            labels[file_name].append((bottom, left, top, right))
    return labels, len(lines)


def show_image(image, label="Image"):
    fig, (ax1) = pyplot.subplots(1, 1, figsize=(8, 4), sharex='all', sharey='all')
    ax1.axis('off')
    ax1.imshow(image, cmap=pyplot.cm.gray)
    ax1.set_title(label)
    pyplot.show()
