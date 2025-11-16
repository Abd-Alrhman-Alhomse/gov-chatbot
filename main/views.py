from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .docs_handling import docHandling
from .llm import chatBot
from .models import question
class AskView(APIView):
    def post(self, request):
        user_message = request.data.get('message', '')
        if not user_message:
            return Response({'error': 'Message is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            ch = chatBot()
            bot_response = ch.chat(user_message)
            print('here')
            ip = request.META.get('REMOTE_ADDR')
            q = question.objects.create(user_address=ip, user_question=user_message, model_answer=bot_response)
            q.save()
        except:
            return Response({'error': 'error happend while handiling the message'}, status=status.HTTP_400_BAD_REQUEST)

        return Response({'response': bot_response['answer']})



class AddDocView(APIView):
    def post(self, request):
        document = request.data.get('document')
        if not document:
            return Response({'error': 'document is required'}, status=status.HTTP_400_BAD_REQUEST)
    
        try:
            print('1')
            docs = docHandling()
            print('2')
            docs.add_document(document)
            print('3')
        except:
             return Response({'error': 'error happened while handle the document'}, status=status.HTTP_400_BAD_REQUEST)

        return Response({'success': 'true'})