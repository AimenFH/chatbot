from openai import OpenAI

openai = OpenAI(api_key='sk-54YzuGKKeUx4sgt50MpsA_FxPkGE0NJslg2ZS8GCF-T3BlbkFJ35Z35ntUZcDJSNTsDGH_x8fhIaVaTpCUr8Jcn5-AwA')

response = openai.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[{
        'role': 'system',
        'content': 'you are a helpful assistant.'
    },
        {
            'role': 'user',
            'content': 'write me a 3 paragraph bio',
        }
    ],
    temperature=.6,

)

print(response)