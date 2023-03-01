import random


class RandomSelector:

    def select_next_batch(self, model, training_set, selection_count):
        scores = []
        for i in range(len(training_set.remaining_image_paths)):
            scores.append(random.random())
        if not len(scores) == 0:
            selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        else:
            selected_samples = selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=True)))[1]
        training_set.expand_training_set(selected_samples)

    def ranking(self, model, training_set):
        scores = []
        for i in range(len(training_set.remaining_image_paths)):
            scores.append(random.random())

        
        selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=True)))[1]
        return selected_samples
