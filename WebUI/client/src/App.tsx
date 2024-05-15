import { useState } from 'react'
import './App.css'
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from '@/components/ui/label';
import {
  Card,
} from "@/components/ui/card"


function App() {
  const [input, setInput] = useState<string>("");
  const [result, setResult] = useState<string>("");

  const submit = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: input }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch');
      }

      const data = await response.json();
      console.log(data.label);
      setResult(data.label);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const clear = () => {
    setResult("");
  }

  return (
    <div className="container mx-auto">
      <div className='flex justify-center items-center h-screen px-32'>
        <Card className='w-full p-8'>
            <div className='p-2 flex justify-center items-center'>
                {result && <Label>This text is {result}</Label>}
            </div>
            <Textarea className="min-h-60" value={input} onChange={(e) => setInput(e.target.value)} placeholder='Type here....'></Textarea>
            <Button className="mt-4" onClick={submit}>Submit</Button>
            <Button className="mx-4 bg-white text-black outline" onClick={clear}>Clear</Button>
        </Card>
      </div>
    </div>
  )
}

export default App
