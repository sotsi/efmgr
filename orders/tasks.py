from celery import shared_task
from django.core.mail import send_mail
from .models import Order

@shared_task
def order_created(order_id):
	"""
	Task to send an e-mail notification when an order is
	successfully created.
	"""
	order = Order.objects.get(id=order_id)
	subject = 'Order nr. {}'.format(order.id)
	message = 'Dear {},\n\nYou have successfully placed an order.\Your order id is {}.'.format(order.first_name,order.id)
	# mail_sent = send_mail(subject,message,'so.tsivras@gmail.com',[order.email])
	return message