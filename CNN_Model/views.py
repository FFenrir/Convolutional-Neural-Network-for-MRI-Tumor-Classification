import os
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .prediction import MRIModelPrediction  

class PredictionAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        image_file = request.data.get('image')

        if not image_file:
            return Response({'error': 'No image provided'}, status=400)

        # Save the uploaded image temporarily
        temp_image_path = 'temp_image.jpg'
        with open(temp_image_path, 'wb') as f:
            f.write(image_file.read())

        # Make predictions using your predictor
        model_predictor = MRIModelPrediction()
        prediction_result = model_predictor.make_prediction(temp_image_path)

        # Optionally, save predictions to your database
        # prediction_instance = Prediction(label=prediction_result['label'], probability=prediction_result['probability'])
        # prediction_instance.save()

        # Delete the temporary image
        os.remove(temp_image_path)

        # Return predictions as JSON
        return Response(prediction_result, status=200)