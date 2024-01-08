def Inference(image_path):
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.axis('off')  
    plt.title('Input Image')
    plt.show()
    image = transform(image).unsqueeze(0).to(device)
    
    saved_model_path = 'model.pth'  
    model = torch.load(saved_model_path)
    model.eval() 
    with torch.no_grad():
        age_pred,gender_pred,race_pred = model(image)
    age_pred=torch.round(age_pred).float()
    gender_pred=torch.round(gender_pred).float()
    race_probabilities = F.softmax(race_pred, dim=1).cpu().numpy()
    race_pred=np.argmax(race_probabilities,axis=1)
    gender_map={0.0:'Male',1.0:'Female'}
    race_map={0.0:'White',1.0:'Black',2.0:'Asian',3.0:'Indian',4.0:'Others'}
    print(f"Age:{age_pred.item()},\nGender:{gender_map[gender_pred.item()]}, \nRace:{race_map[race_pred[0]]}")
    
