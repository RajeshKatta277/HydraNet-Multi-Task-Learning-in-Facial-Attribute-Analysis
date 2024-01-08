class Hydranet(nn.Module):
    def __init__(self):
        super(Hydranet,self).__init__()
        self.backbone=models.resnet34(pretrained=True)
        self.model_in_features=self.backbone.fc.in_features
        self.backbone.fc=nn.Identity()
        self.backbone.fc1=nn.Sequential(nn.Linear(self.model_in_features,self.model_in_features),nn.ELU(),nn.Linear(self.model_in_features,1))
        self.backbone.fc2=nn.Sequential(nn.Linear(self.model_in_features,self.model_in_features),nn.ELU(),nn.Linear(self.model_in_features,1),nn.Sigmoid())
        self.backbone.fc3=nn.Sequential(nn.Linear(self.model_in_features,self.model_in_features),nn.ELU(),nn.Linear(self.model_in_features,5))
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self,x):
        age_head = self.backbone.fc1(self.backbone(x))
        gender_head = self.backbone.fc2(self.backbone(x))
        race_head = self.backbone.fc3(self.backbone(x))
        return age_head, gender_head, race_head
