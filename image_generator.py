from PIL import Image, ImageDraw
import random


class ImageGenerator:
    def __init__(self, image_size, object_path):
        self.height = image_size[0]
        self.width = image_size[1]
        self.object_path = object_path
        self.objects = [] # list of tuples ((center_x, center_y), radius)
        
    def are_close(self, new_center, new_radius):
        for (center, radius) in self.objects:
            dist = ((center[0] - new_center[0]) ** 2 + (center[1] - new_center[1]) ** 2) ** 0.5
            
            if dist < (radius + new_radius + min(self.height, self.width) // 30):
                return True
            
        return False

    def generate_image(self, num_targets):
        background = Image.new('RGBA', (self.height, self.width), (220, 220, 220, 255))
        target_image = Image.open(self.object_path).convert("RGBA")
        
        min_radius = min(self.height, self.width) // 35
        max_radius = min(self.height, self.width) // 15
        radius = min(self.height, self.width) // 25
        
        def place_object():
            max_attempts = 100  # to prevent infinite loops if no space is left
            
            for _ in range(max_attempts):
                # uncomment next line to make targets of different size
                # radius = random.randint(min_radius, max_radius)
                x = random.uniform(radius, self.height - radius)
                y = random.uniform(radius, self.width - radius)
                
                if not self.are_close((x, y), radius):
                    resized_target = target_image.resize((radius * 2, radius * 2))
                    background.paste(resized_target, (int(x - radius), int(y - radius)), resized_target)
                    
                    self.objects.append(((x, y), radius))
                    return True
                
            return False

        for _ in range(num_targets):
            place_object()

        output_path = r'Final-Project/Project-1_Hitting_Target/assets/generated_image.png'
        background.save(output_path)
        background.show()
        return output_path