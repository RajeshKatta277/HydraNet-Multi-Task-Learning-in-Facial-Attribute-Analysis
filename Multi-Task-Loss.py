class MultiTaskLoss(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLoss, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, losses):
        weighted_losses = torch.exp(-self.log_vars) * losses
        return torch.sum(weighted_losses)
