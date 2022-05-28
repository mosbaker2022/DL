import torch

# Training Function
def train(model, optimizer, device, criterion, train_loader, valid_loader, num_epochs):
    # initialize running values
    best_valid_loss = float("Inf")
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    #global_steps_list = []
    step = 0

    # training loop
    model.train()
    for epoch in range(num_epochs):
        if epoch == 20:
            xx=1
        running_loss = 0.0
        model.train()
        train_accuracy = 0
        train_count = 0
        for (data, labels, data_len) in train_loader:
            labels = labels.to(device)
            data = data.to(device)
            data_len = data_len.to('cpu')
            output = model(data, data_len)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_accuracy += torch.sum(torch.argmax(output,1) == labels)
            train_count += len(labels)

            # update running values
            running_loss += loss.item()
            step += 1

            # evaluation step

        # After each training epoch we run evaluation
        model.eval()
        with torch.no_grad():
            test_accuracy = 0
            test_count = 0
            valid_running_loss = 0.0
            # validation loop
            for (data, labels, data_len) in valid_loader:
                labels = labels.to(device)
                data = data.to(device)
                data_len = data_len.to('cpu')
                output = model(data, data_len)
                loss = criterion(output, labels)
                valid_running_loss += loss.item()
                test_accuracy += torch.sum(torch.argmax(output,1) == labels)
                test_count += len(labels)

        # evaluation
        average_train_loss = running_loss / len(train_loader)
        average_valid_loss = valid_running_loss / len(valid_loader)
        train_loss_list.append(average_train_loss)
        valid_loss_list.append(average_valid_loss)
        average_train_accuracy = train_accuracy / train_count
        average_test_accuracy = test_accuracy / test_count
        train_accuracy_list.append(average_train_accuracy)
        test_accuracy_list.append(average_test_accuracy)

        # print progress
        print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f},  Train Acc: {:.4f}, Valid Acc: {:.4f},'
              .format(epoch + 1, num_epochs,
                      average_train_loss, average_valid_loss, average_train_accuracy, average_test_accuracy))

        # checkpoint
        if best_valid_loss > average_valid_loss:
            best_valid_loss = average_valid_loss
            #print('found better model')
    return [train_loss_list, valid_loss_list, train_accuracy_list, test_accuracy_list]
