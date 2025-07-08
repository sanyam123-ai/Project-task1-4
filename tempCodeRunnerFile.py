    style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim the model
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i + 1]

    return model, style_losses, content_losses

# Optimization
def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300, style_weight=1e6, content_weight=1):
    print("🔄 Optimizing...")
    model, style_losses, content_losses = get_model_and_losses(cnn, content_img, style_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()

            if run[0] % 50 == 0:
                print(f"Step {run[0]}, Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")
            run[0] += 1
            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# Main block
if __name__ == "__main__":
    # Paths to your images
    content_path = "content.jpg"
    style_path = "style.jpg"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load content and style images
    content = load_image(content_path)
    style = load_image(style_path)

    # Start with a copy of the content image
    input_img = content.clone()

    # Load VGG19
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Run style transfer
    output = run_style_transfer(cnn, content, style, input_img)

    # Save and show the result
    styled_image = im_convert(output)
    styled_image.save("stylized_output.jpg")
    styled_image.show()